"""Shared pytest configuration.

``db/db.py`` resolves ``DB_FILE`` from the environment **at import time**, so the
redirection below has to happen before any test module imports ``app`` or
``db.db``. Without it the suite reads and writes the real
``data/selenium_engine.db`` and mutates operator state — notably the persisted
default engine.
"""

import os
import sqlite3
import tempfile
from pathlib import Path

_TEST_DB = Path(tempfile.gettempdir()) / "selenium_llm_engine_tests.db"
# Always override: a test run must never reach the production database.
os.environ["SELENIUM_LLM_DB"] = str(_TEST_DB)
_TEST_DB.unlink(missing_ok=True)

# ``app`` installs a file log handler at import time; keep it out of the
# working tree.
os.environ["SELENIUM_LOG_DIR"] = str(Path(tempfile.gettempdir()) / "selenium_llm_engine_test_logs")

import pytest  # noqa: E402  (import after the environment is prepared)

from db.db import init_database  # noqa: E402

# The suite previously leaned on the production database already having its
# schema; create it explicitly in the throwaway one.
init_database()


def _clear_persisted_default() -> None:
    """Drop the persisted default engine and the manager's cached copy."""
    conn = sqlite3.connect(str(_TEST_DB))
    try:
        conn.execute("DELETE FROM settings WHERE key = 'default_engine'")
        conn.commit()
    except sqlite3.Error:
        # The table only exists once init_database() has run.
        pass
    finally:
        conn.close()

    from core.engine_manager import EngineManager

    manager = EngineManager._instance
    if manager is not None:
        manager.default_engine = None


@pytest.fixture(autouse=True)
def isolate_default_engine():
    """Keep the default engine from leaking between tests.

    ``set_default_engine`` persists its choice, so without this reset one test's
    default would survive into the next and make the suite order-dependent.
    """
    _clear_persisted_default()
    yield
    _clear_persisted_default()
