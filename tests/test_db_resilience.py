"""Regression tests for db.db write resilience.

A full or read-only SQLite database previously raised
``sqlite3.OperationalError`` (e.g. "database or disk is full") from the
best-effort telemetry writes (``inc_requests``/``log_prompt``), which turned
every prompt request into an HTTP 500 and made the engine appear unresponsive.

These tests assert that stat counters and prompt logging degrade gracefully
(swallow the operational error, keep serving) instead of crashing the caller,
while read paths keep working.
"""

import sqlite3

import pytest

import db.db as dbmod


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Point the module at a fresh, initialised on-disk database."""
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(dbmod, "DB_FILE", db_file)
    dbmod.init_database()
    return db_file


class _DiskFullCursor(sqlite3.Cursor):
    def execute(self, sql, *args, **kwargs):  # type: ignore[override]
        if sql.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            raise sqlite3.OperationalError("database or disk is full")
        return super().execute(sql, *args, **kwargs)


class _DiskFullConnection(sqlite3.Connection):
    def cursor(self, *args, **kwargs):  # type: ignore[override]
        return super().cursor(factory=_DiskFullCursor)


def _force_disk_full(monkeypatch):
    """Make every write cursor.execute raise the 'disk is full' error."""
    real_connect = sqlite3.connect

    def fake_connect(*args, **kwargs):
        kwargs["factory"] = _DiskFullConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(dbmod.sqlite3, "connect", fake_connect)


def test_inc_stat_does_not_raise_when_disk_full(temp_db, monkeypatch):
    _force_disk_full(monkeypatch)
    # None of these must raise, even though the underlying writes fail.
    dbmod.inc_requests()
    dbmod.inc_responses()
    dbmod.inc_errors()
    dbmod.inc_media_sent(2)


def test_log_prompt_does_not_raise_when_disk_full(temp_db, monkeypatch):
    _force_disk_full(monkeypatch)
    dbmod.log_prompt("my-engine", "default", "hi", "there", "success", 123)


def test_reads_still_work_after_failed_writes(temp_db, monkeypatch):
    # A successful write first, then simulate the disk filling up.
    dbmod.inc_requests()
    _force_disk_full(monkeypatch)
    dbmod.inc_requests()  # swallowed
    # Reads must keep functioning and reflect the earlier successful write.
    stats = dbmod.get_stats()
    assert stats.get("requests", 0) >= 1


def test_normal_writes_persist(temp_db):
    dbmod.inc_requests()
    dbmod.inc_requests()
    dbmod.log_prompt("my-engine", "default", "p", "r", "success", 10)
    assert dbmod.get_stats().get("requests") == 2
    logs = dbmod.get_prompt_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["engine"] == "my-engine"
