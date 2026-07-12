import logging
import os
import sqlite3
import threading
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DB_FILE = Path(os.getenv("SELENIUM_LLM_DB", "./data/selenium_engine.db"))
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

DB_LOCK = threading.Lock()

CREATE_PROMPT_LOGS = """
CREATE TABLE IF NOT EXISTS prompt_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    engine TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    status TEXT NOT NULL,
    elapsed_ms INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_STATS = """
CREATE TABLE IF NOT EXISTS stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value INTEGER NOT NULL
)
"""


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(CREATE_PROMPT_LOGS)
            cur.execute(CREATE_STATS)
            conn.commit()
        finally:
            conn.close()


def log_prompt(
    engine: str, model: str, prompt: str, response: str, status: str, elapsed_ms: int
) -> None:
    # Persisting a prompt log is best-effort: a full or read-only DB must never
    # crash the request that produced a valid response.
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO prompt_logs (engine, model, prompt, response, status, elapsed_ms) VALUES (?, ?, ?, ?, ?, ?)",
                (engine, model, prompt, response, status, elapsed_ms),
            )
            conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning("log_prompt skipped, DB write failed: %s", e)
        finally:
            conn.close()


def get_prompt_logs(
    limit: int = 100,
    offset: int = 0,
    engine: str | None = None,
    model: str | None = None,
    status: str | None = None,
) -> List[Dict[str, Any]]:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            query = "SELECT * FROM prompt_logs"
            where_clauses = []
            params: list[Any] = []

            if engine:
                where_clauses.append("engine = ?")
                params.append(engine)
            if model:
                where_clauses.append("model = ?")
                params.append(model)
            if status:
                where_clauses.append("status = ?")
                params.append(status)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            return [dict(x) for x in rows]
        finally:
            conn.close()


def _inc_stat(key: str, amount: int = 1) -> None:
    # Counters are best-effort telemetry: a full or read-only DB must never
    # crash the request being served (previously this raised sqlite3
    # OperationalError "database or disk is full" and turned every prompt into
    # an HTTP 500).
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO stats (key, value) VALUES (?, ?)", (key, 0)
            )
            cur.execute(
                "UPDATE stats SET value = value + ? WHERE key = ?", (amount, key)
            )
            conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning("stat '%s' increment skipped, DB write failed: %s", key, e)
        finally:
            conn.close()


def inc_requests() -> None:
    _inc_stat("requests", 1)


def inc_responses() -> None:
    _inc_stat("responses", 1)


def inc_errors() -> None:
    _inc_stat("errors", 1)


def _today_media_key() -> str:
    return f"media_sent:{date.today().isoformat()}"


def inc_media_sent(amount: int = 1) -> None:
    if amount <= 0:
        return
    _inc_stat(_today_media_key(), amount)


def get_media_sent_today() -> int:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT value FROM stats WHERE key = ?", (_today_media_key(),))
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()


def get_stats() -> Dict[str, int]:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT key, value FROM stats")
            return {row[0]: row[1] for row in cur.fetchall()}
        finally:
            conn.close()


def clear_stats() -> None:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            try:
                cur.execute("DELETE FROM stats")
                conn.commit()
            except sqlite3.OperationalError as e:
                # Some environments may use a readonly DB; ignore if clearing stats fails.
                # The UI still resets counters locally and the app remains functional.
                print(f"Warning: cannot clear stats DB ({e})")
        finally:
            conn.close()


def clear_prompt_logs() -> None:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            try:
                cur.execute("DELETE FROM prompt_logs")
                conn.commit()
            except sqlite3.OperationalError as e:
                # Some environments may use a readonly DB; ignore if clearing logs fails.
                print(f"Warning: cannot clear prompt_logs DB ({e})")
        finally:
            conn.close()


def get_response_time_stats() -> Dict[str, Any]:
    """Return average elapsed_ms per engine and global average in milliseconds."""
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT AVG(elapsed_ms) AS avg_ms FROM prompt_logs WHERE elapsed_ms IS NOT NULL"
            )
            row = cur.fetchone()
            global_avg = row[0] if row and row[0] is not None else None

            cur.execute(
                "SELECT engine, AVG(elapsed_ms) AS avg_ms, COUNT(*) AS count "
                "FROM prompt_logs WHERE elapsed_ms IS NOT NULL GROUP BY engine ORDER BY engine"
            )
            per_engine = {
                r[0]: float(r[1]) if r[1] is not None else None for r in cur.fetchall()
            }

            return {
                "global_avg_ms": float(global_avg) if global_avg is not None else None,
                "per_engine_avg_ms": per_engine,
            }
        finally:
            conn.close()


def get_logged_engines() -> List[str]:
    """Return the list of distinct engine names that have at least one prompt log."""
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT engine FROM prompt_logs ORDER BY engine")
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()
