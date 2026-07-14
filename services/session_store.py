"""SQLite-backed session store.

Persistent across service restarts.  Uses a single SQLite database file;
WAL mode is enabled for good concurrent read/write performance under
FastAPI's async event loop.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model (same as before, purely in-memory representation)
# ---------------------------------------------------------------------------


@dataclass
class Session:
    session_id: str
    user_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    return_order_id: str = ""
    return_reason: str = ""
    return_step: str = ""
    need_handoff: bool = False
    handoff_reason: str = ""
    created_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.created_at = time.time()


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    messages     TEXT NOT NULL DEFAULT '[]',   -- JSON array
    return_order_id  TEXT NOT NULL DEFAULT '',
    return_reason    TEXT NOT NULL DEFAULT '',
    return_step      TEXT NOT NULL DEFAULT '',
    need_handoff     INTEGER NOT NULL DEFAULT 0,  -- boolean
    handoff_reason   TEXT NOT NULL DEFAULT '',
    created_at       REAL NOT NULL              -- unix timestamp (float)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_handoff ON sessions(need_handoff);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SessionStore:
    """SQLite-backed session container.

    Thread-safe: all writes are serialised through a ``threading.Lock``.
    Connections are created per-operation (the GIL + WAL mode make this
    safe for typical FastAPI workloads).
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = settings.db_path
        self._db_path = Path(db_path)
        self._lock = threading.Lock()

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Bootstrap tables (executescript for multi-statement DDL)
        self._executescript(SCHEMA_SQL)

    # ── low-level helpers ──────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Return a new connection for the current thread / task."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _executescript(self, sql: str) -> None:
        """Run multi-statement DDL (CREATE TABLE, CREATE INDEX, etc.)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(sql)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _execute(
        self, sql: str, params: tuple | None = None, *, write: bool = False
    ) -> list[sqlite3.Row]:
        """Execute SQL and return fetched rows."""
        with self._lock if write else _noop_context():
            conn = self._connect()
            try:
                cur = conn.execute(sql, params or ())
                if write:
                    conn.commit()
                rows = cur.fetchall()
                return rows
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ── public API ─────────────────────────────────────────────────────────

    def create(self, user_id: str) -> Session:
        sid = uuid.uuid4().hex[:16]
        session = Session(session_id=sid, user_id=user_id)
        now = time.time()

        self._execute(
            """INSERT INTO sessions (session_id, user_id, messages, created_at)
               VALUES (?, ?, '[]', ?)""",
            (sid, user_id, now),
            write=True,
        )
        logger.info("Created session %s for user %s", sid, user_id)
        return session

    def get(self, session_id: str) -> Session | None:
        rows = self._execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        if not rows:
            return None

        row = rows[0]

        # TTL eviction
        age = time.time() - row["created_at"]
        if age > settings.session_ttl_minutes * 60:
            self.delete(session_id)
            logger.info("Session %s expired (age=%.0fs)", session_id, age)
            return None

        # Touch (update heartbeat)
        self._execute(
            "UPDATE sessions SET created_at = ? WHERE session_id = ?",
            (time.time(), session_id),
            write=True,
        )

        return self._row_to_session(row)

    def save(self, session: Session) -> None:
        self._execute(
            """UPDATE sessions SET
                 user_id = ?, messages = ?, return_order_id = ?,
                 return_reason = ?, return_step = ?, need_handoff = ?,
                 handoff_reason = ?, created_at = ?
               WHERE session_id = ?""",
            (
                session.user_id,
                json.dumps(session.messages, ensure_ascii=False),
                session.return_order_id,
                session.return_reason,
                session.return_step,
                int(session.need_handoff),
                session.handoff_reason,
                session.created_at,
                session.session_id,
            ),
            write=True,
        )

    def delete(self, session_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?", (session_id,)
                )
                conn.commit()
                deleted = cur.rowcount > 0
                if deleted:
                    logger.info("Deleted session %s", session_id)
                return deleted
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    @property
    def pending_handoffs(self) -> list[Session]:
        rows = self._execute(
            "SELECT * FROM sessions WHERE need_handoff = 1"
        )
        return [self._row_to_session(r) for r in rows]

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        messages_raw = row["messages"]
        try:
            messages = json.loads(messages_raw)
        except (json.JSONDecodeError, TypeError):
            messages = []

        return Session(
            session_id=row["session_id"],
            user_id=row["user_id"],
            messages=messages,
            return_order_id=row["return_order_id"],
            return_reason=row["return_reason"],
            return_step=row["return_step"],
            need_handoff=bool(row["need_handoff"]),
            handoff_reason=row["handoff_reason"],
            created_at=row["created_at"],
        )


# ── contextlib helper ───────────────────────────────────────────────────────

class _noop_context:
    """A no-op context manager used when we don't need the lock."""

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> None:
        pass


# ── global singleton ─────────────────────────────────────────────────────────

session_store = SessionStore()
