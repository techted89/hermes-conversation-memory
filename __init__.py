"""Conversation Memory Provider — MemoryProvider interface.

Local SQLite-backed conversation memory: persistent turn/session indexing with FTS5 search,
explicit recall/save tools, and automatic background persistence via sync_turn/on_pre_compress/on_session_end.

Config (memory.conversation_memory in config.yaml):
  - max_context_chars: int (default 8000) — max chars per prefetch injection
  - max_turns_per_session: int (default 1000) — retention cap per session
  - fts_enabled: bool (default true) — enable FTS5 full-text search
  - auto_extract: bool (default true) — mirror built-in memory writes & persist turns

No required env vars. Working directory: $HERMES_HOME/conversation_memory/ (profile-scoped).
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import (
    MemoryProvider,
    RecallStatus,
    spawn_context_thread,
    is_trivial_prompt,
)
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Helpers
# ──────────────────────────────────────────────────────────────────────────────

_DB_NAME = "conversation_memory.db"
_FTS_TABLE = "turns_fts"
_TURNS_TABLE = "turns"
_SESSIONS_TABLE = "sessions"
_DEFAULT_MAX_CONTEXT = 8000
_DEFAULT_MAX_TURNS = 1000
_MIN_QUERY_LEN = 3
_TURN_BATCH_SIZE = 50

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_SESSIONS_TABLE} (
    session_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turn_count INTEGER DEFAULT 0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS {_TURNS_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (session_id) REFERENCES {_SESSIONS_TABLE}(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON {_TURNS_TABLE}(session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON {_TURNS_TABLE}(timestamp);
"""

_FTS_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
    session_id UNINDEXED,
    turn_number UNINDEXED,
    role UNINDEXED,
    content,
    timestamp UNINDEXED,
    content={_TURNS_TABLE},
    content_rowid=id
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS {_FTS_TABLE}_ai AFTER INSERT ON {_TURNS_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}(rowid, session_id, turn_number, role, content, timestamp)
    VALUES (new.id, new.session_id, new.turn_number, new.role, new.content, new.timestamp);
END;

CREATE TRIGGER IF NOT EXISTS {_FTS_TABLE}_ad AFTER DELETE ON {_TURNS_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, session_id, turn_number, role, content, timestamp)
    VALUES ('delete', old.id, old.session_id, old.turn_number, old.role, old.content, old.timestamp);
END;

CREATE TRIGGER IF NOT EXISTS {_FTS_TABLE}_au AFTER UPDATE ON {_TURNS_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, session_id, turn_number, role, content, timestamp)
    VALUES ('delete', old.id, old.session_id, old.turn_number, old.role, old.content, old.timestamp);
    INSERT INTO {_FTS_TABLE}(rowid, session_id, turn_number, role, content, timestamp)
    VALUES (new.id, new.session_id, new.turn_number, new.role, new.content, new.timestamp);
END;
"""

_WRITE_QUEUE_SENTINEL = object()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_config() -> dict[str, Any]:
    """Read memory.conversation_memory from config.yaml."""
    try:
        from hermes_cli.config import load_config_readonly
        block = load_config_readonly().get("memory", {}).get("conversation_memory", {})
    except Exception:
        block = None
    return dict(block) if isinstance(block, dict) else {}


def _schema_to_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tool Schemas
# ──────────────────────────────────────────────────────────────────────────────

CONVERSATION_SEARCH_SCHEMA = _schema_to_tool(
    "conversation_memory_search",
    "Search conversation history across all sessions using full-text search. Returns ranked turns with session context.",
    {
        "query": {"type": "string", "description": "Search query (keywords, phrases)."},
        "session_id": {"type": "string", "description": "Optional: restrict to a specific session."},
        "limit": {"type": "integer", "description": "Max results (default 10, max 50).", "default": 10},
        "role": {"type": "string", "description": "Optional: filter by role (user, assistant, tool).", "enum": ["user", "assistant", "tool"]},
    },
    ["query"],
)

CONVERSATION_RECENT_SCHEMA = _schema_to_tool(
    "conversation_memory_recent",
    "Get the most recent conversation turns from a session (or the current session if omitted).",
    {
        "session_id": {"type": "string", "description": "Optional: session to retrieve (default: current session)."},
        "limit": {"type": "integer", "description": "Max turns to return (default 20, max 100).", "default": 20},
    },
    [],
)

CONVERSATION_SAVE_SCHEMA = _schema_to_tool(
    "conversation_memory_save",
    "Explicitly save an important fact, decision, or note to conversation memory with optional tags.",
    {
        "content": {"type": "string", "description": "The information to remember."},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization."},
        "importance": {"type": "number", "description": "Importance 0.0-1.0 (default 0.7).", "default": 0.7},
    },
    ["content"],
)

CONVERSATION_STATUS_SCHEMA = _schema_to_tool(
    "conversation_memory_status",
    "Show conversation memory database status: sessions, turns, disk size, FTS state.",
    {},
    [],
)

CONVERSATION_CLEAR_SCHEMA = _schema_to_tool(
    "conversation_memory_clear",
    "Delete all conversation memory for a session (or all sessions if session_id omitted). Use with caution.",
    {
        "session_id": {"type": "string", "description": "Optional: session to clear (default: all sessions)."},
        "confirm": {"type": "boolean", "description": "Must be true to confirm deletion.", "default": False},
    },
    [],
)

_ALL_SCHEMAS = [
    CONVERSATION_SEARCH_SCHEMA,
    CONVERSATION_RECENT_SCHEMA,
    CONVERSATION_SAVE_SCHEMA,
    CONVERSATION_STATUS_SCHEMA,
    CONVERSATION_CLEAR_SCHEMA,
]

_TOOL_NAMES = {s["name"] for s in _ALL_SCHEMAS}


# ──────────────────────────────────────────────────────────────────────────────
# Database Layer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _TurnRecord:
    id: int
    session_id: str
    turn_number: int
    role: str
    content: str
    timestamp: str
    metadata_json: Optional[str]


class _ConversationDB:
    """Thread-local SQLite connection with FTS5, write queue, and retention."""

    def __init__(self, db_path: Path, max_turns_per_session: int, fts_enabled: bool):
        self._db_path = db_path
        self._max_turns = max_turns_per_session
        self._fts_enabled = fts_enabled
        self._local = threading.local()
        self._queue: "queue.Queue[tuple | object]" = __import__("queue").Queue()
        self._writer_thread = spawn_context_thread(self._writer_loop, name="convmem-writer")
        self._writer_thread.start()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = sqlite3.connect(
                str(self._db_path), timeout=30, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        if self._fts_enabled:
            try:
                conn.executescript(_FTS_SCHEMA)
            except sqlite3.OperationalError as e:
                logger.warning("FTS5 unavailable, disabling: %s", e)
                self._fts_enabled = False
        conn.commit()

    def _writer_loop(self) -> None:
        """Background writer: batches turn inserts, applies retention."""
        batch: list[tuple] = []
        last_flush = time.monotonic()
        FLUSH_INTERVAL = 1.0  # flush at least once per second
        while True:
            # Use timeout to allow periodic flush
            try:
                item = self._queue.get(timeout=0.1)
            except Exception:
                # Timeout - check if we should flush due to time
                if batch and time.monotonic() - last_flush >= FLUSH_INTERVAL:
                    try:
                        self._flush_batch(batch)
                    except sqlite3.OperationalError as e:
                        if "unable to open database file" in str(e):
                            logger.debug("Database file gone, exiting writer loop")
                            break
                        logger.error("Conversation memory flush failed: %s", e)
                    except Exception as e:
                        logger.error("Conversation memory flush failed: %s", e)
                    batch.clear()
                    last_flush = time.monotonic()
                continue

            if item is _WRITE_QUEUE_SENTINEL:
                if batch:
                    try:
                        self._flush_batch(batch)
                    except sqlite3.OperationalError as e:
                        if "unable to open database file" in str(e):
                            logger.debug("Database file gone, skipping final flush")
                        else:
                            logger.error("Conversation memory final flush failed: %s", e)
                    except Exception as e:
                        logger.error("Conversation memory final flush failed: %s", e)
                break
            batch.append(item)
            if len(batch) >= _TURN_BATCH_SIZE:
                try:
                    self._flush_batch(batch)
                except sqlite3.OperationalError as e:
                    if "unable to open database file" in str(e):
                        logger.debug("Database file gone, exiting writer loop")
                        break
                    logger.error("Conversation memory batch flush failed: %s", e)
                except Exception as e:
                    logger.error("Conversation memory batch flush failed: %s", e)
                batch.clear()
                last_flush = time.monotonic()

    def _flush_batch(self, batch: list[tuple]) -> None:
        if not batch:
            return
        conn = self._get_conn()
        try:
            conn.executemany(
                f"INSERT INTO {_TURNS_TABLE} (session_id, turn_number, role, content, timestamp, metadata_json) VALUES (?,?,?,?,?,?)",
                batch,
            )
            # Apply retention per session
            for session_id, _, _, _, _, _ in batch:
                self._apply_retention(conn, session_id)
            conn.commit()
        except Exception as e:
            logger.error("Conversation memory write batch failed: %s", e)
            conn.rollback()

    def _apply_retention(self, conn: sqlite3.Connection, session_id: str) -> None:
        if self._max_turns <= 0:
            return
        # Delete oldest turns beyond the cap
        conn.execute(
            f"DELETE FROM {_TURNS_TABLE} WHERE session_id = ? AND id NOT IN "
            f"(SELECT id FROM {_TURNS_TABLE} WHERE session_id = ? ORDER BY turn_number DESC LIMIT ?)",
            (session_id, session_id, self._max_turns),
        )

    def enqueue_turn(
        self,
        session_id: str,
        turn_number: int,
        role: str,
        content: str,
        timestamp: str,
        metadata_json: Optional[str] = None,
    ) -> None:
        self._queue.put((session_id, turn_number, role, content, timestamp, metadata_json))

    def flush(self, timeout: float = 5.0) -> None:
        """Block until queue drains."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.empty():
                return
            time.sleep(0.01)
        logger.warning("Conversation memory flush timeout")

    def shutdown(self) -> None:
        self._queue.put(_WRITE_QUEUE_SENTINEL)
        # Give the writer thread a moment to process the sentinel
        self._writer_thread.join(timeout=10.0)
        if self._writer_thread.is_alive():
            logger.warning("Conversation memory writer thread did not stop cleanly")
        # Close thread-local connection
        conn = getattr(self._local, "conn", None)
        if conn:
            with contextlib.suppress(Exception):
                conn.close()
            self._local.conn = None

    # ── Read Operations ──

    def ensure_session(self, session_id: str, title: Optional[str] = None) -> None:
        conn = self._get_conn()
        now = _now_iso()
        conn.execute(
            f"INSERT OR IGNORE INTO {_SESSIONS_TABLE} (session_id, title, created_at, updated_at, turn_count) VALUES (?,?,?,?,0)",
            (session_id, title or "", now, now),
        )
        conn.commit()

    def update_session_title(self, session_id: str, title: str) -> None:
        conn = self._get_conn()
        conn.execute(
            f"UPDATE {_SESSIONS_TABLE} SET title = ?, updated_at = ? WHERE session_id = ?",
            (title, _now_iso(), session_id),
        )
        conn.commit()

    def increment_turn_count(self, session_id: str) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            f"UPDATE {_SESSIONS_TABLE} SET turn_count = turn_count + 1, updated_at = ? WHERE session_id = ? RETURNING turn_count",
            (_now_iso(), session_id),
        )
        row = cur.fetchone()
        conn.commit()
        return row["turn_count"] if row else 0

    def get_turn_count(self, session_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(f"SELECT turn_count FROM {_SESSIONS_TABLE} WHERE session_id = ?", (session_id,)).fetchone()
        return row["turn_count"] if row else 0

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        role: Optional[str] = None,
    ) -> list[_TurnRecord]:
        if not self._fts_enabled or len(query.strip()) < _MIN_QUERY_LEN:
            return []
        conn = self._get_conn()
        sql = f"""
            SELECT t.id, t.session_id, t.turn_number, t.role, t.content, t.timestamp, t.metadata_json
            FROM {_TURNS_TABLE} t
            JOIN {_FTS_TABLE} f ON t.id = f.rowid
            WHERE {_FTS_TABLE} MATCH ?
        """
        params: list[Any] = [query]
        if session_id:
            sql += " AND t.session_id = ?"
            params.append(session_id)
        if role:
            sql += " AND t.role = ?"
            params.append(role)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [_TurnRecord(**dict(r)) for r in rows]

    def recent(self, session_id: str, limit: int = 20) -> list[_TurnRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT id, session_id, turn_number, role, content, timestamp, metadata_json "
            f"FROM {_TURNS_TABLE} WHERE session_id = ? ORDER BY turn_number DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [_TurnRecord(**dict(r)) for r in reversed(rows)]

    def list_sessions(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT session_id, title, created_at, updated_at, turn_count FROM {_SESSIONS_TABLE} ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> int:
        conn = self._get_conn()
        cur = conn.execute(f"DELETE FROM {_SESSIONS_TABLE} WHERE session_id = ?", (session_id,))
        conn.commit()
        return cur.rowcount

    def delete_all_sessions(self) -> int:
        conn = self._get_conn()
        cur = conn.execute(f"DELETE FROM {_SESSIONS_TABLE}")
        conn.commit()
        return cur.rowcount

    def get_stats(self) -> dict:
        conn = self._get_conn()
        sessions = conn.execute(f"SELECT COUNT(*) as c FROM {_SESSIONS_TABLE}").fetchone()["c"]
        turns = conn.execute(f"SELECT COUNT(*) as c FROM {_TURNS_TABLE}").fetchone()["c"]
        db_size = self._db_path.stat().st_size if self._db_path.exists() else 0
        fts_ok = self._fts_enabled
        try:
            if fts_ok:
                conn.execute(f"SELECT 1 FROM {_FTS_TABLE} LIMIT 1")
        except Exception:
            fts_ok = False
        return {
            "sessions": sessions,
            "turns": turns,
            "db_size_bytes": db_size,
            "fts_enabled": fts_ok,
            "db_path": str(self._db_path),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Provider Implementation
# ──────────────────────────────────────────────────────────────────────────────

class ConversationMemoryProvider(MemoryProvider):
    """SQLite-backed conversation memory with FTS5 search and background persistence."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self._config = dict(config) if config is not None else _load_config()
        self._max_context = _coerce_int(self._config.get("max_context_chars"), _DEFAULT_MAX_CONTEXT)
        self._max_turns = _coerce_int(self._config.get("max_turns_per_session"), _DEFAULT_MAX_TURNS)
        self._fts_enabled = _coerce_bool(self._config.get("fts_enabled"), True)
        self._auto_extract = _coerce_bool(self._config.get("auto_extract"), True)

        self._db: Optional[_ConversationDB] = None
        self._session_id = ""
        self._user_id = "default"
        self._turn_count = 0
        self._prefetch_cache = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._recall_status: Optional[RecallStatus] = None
        self._shutdown = False

    @property
    def name(self) -> str:
        return "conversation_memory"

    def is_available(self) -> bool:
        # SQLite is always available in stdlib
        return True

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "max_context_chars",
                "description": "Maximum characters injected per prefetch (default 8000)",
                "type": "integer",
                "default": str(_DEFAULT_MAX_CONTEXT),
                "minimum": 1000,
                "maximum": 50000,
            },
            {
                "key": "max_turns_per_session",
                "description": "Retention cap: max turns kept per session (default 1000, 0 = unlimited)",
                "type": "integer",
                "default": str(_DEFAULT_MAX_TURNS),
                "minimum": 0,
                "maximum": 100000,
            },
            {
                "key": "fts_enabled",
                "description": "Enable FTS5 full-text search (requires SQLite FTS5 compile-in)",
                "type": "boolean",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "auto_extract",
                "description": "Automatically persist turns, mirror built-in memory writes, and run pre-compression flush",
                "type": "boolean",
                "default": "true",
                "choices": ["true", "false"],
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        """Bind to profile-scoped storage and initialize database."""
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            from hermes_constants import get_hermes_home
            hermes_home = str(get_hermes_home())

        self._session_id = session_id or ""
        self._user_id = kwargs.get("user_id", "default") or "default"
        self._turn_count = 0

        db_dir = Path(hermes_home) / "conversation_memory"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / _DB_NAME

        self._db = _ConversationDB(db_path, self._max_turns, self._fts_enabled)

        if self._session_id:
            self._db.ensure_session(self._session_id)

        # Resume turn count from existing session
        if self._session_id:
            self._turn_count = self._db.get_turn_count(self._session_id)

        logger.debug(
            "ConversationMemoryProvider initialized: session=%s, db=%s, turns=%d",
            self._session_id,
            db_path,
            self._turn_count,
        )

    def system_prompt_block(self) -> str:
        if not self._db:
            return ""
        return (
            "# Conversation Memory\n"
            "Active. Persistent local SQLite store with FTS5 search.\n"
            "Use conversation_memory_search to find past context, "
            "conversation_memory_recent for recent turns, "
            "conversation_memory_save to store important facts explicitly."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached prefetch result (populated by queue_prefetch)."""
        if is_trivial_prompt(query) or not self._db:
            self._recall_status = None
            return ""

        with self._prefetch_lock:
            cached = self._prefetch_cache
            self._prefetch_cache = ""

        if not cached:
            self._recall_status = None
            return ""

        # Trim to max_context
        if len(cached) > self._max_context:
            cached = cached[: self._max_context] + "\n\n[... truncated]"

        self._recall_status = RecallStatus(
            provider_label="conversation_memory",
            count=1,
            glyph="🧠",
        )
        return f"## Conversation Memory Recall\n{cached}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background search for next turn's prefetch."""
        if is_trivial_prompt(query) or not self._db:
            return

        sid = session_id or self._session_id
        if not sid:
            return

        # Wait for previous prefetch thread to avoid pileup
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
            if self._prefetch_thread.is_alive():
                logger.debug("Previous prefetch still running; skipping new query")
                return

        def _do_prefetch():
            try:
                results = self._db.search(query, session_id=sid, limit=5)
                if not results:
                    return
                lines = []
                for r in results:
                    role_tag = "User" if r.role == "user" else "Assistant"
                    content = r.content[:500]
                    lines.append(f"[{role_tag}] {content}")
                if lines:
                    with self._prefetch_lock:
                        self._prefetch_cache = "\n".join(lines)
            except Exception as e:
                logger.debug("Prefetch search failed: %s", e)

        self._prefetch_thread = spawn_context_thread(_do_prefetch, name="convmem-prefetch")
        self._prefetch_thread.start()

    def recall_status(self) -> Optional[RecallStatus]:
        return self._recall_status

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[list[dict[str, Any]]] = None,
        turn_author: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist completed turn asynchronously via background writer."""
        if not self._db or not self._auto_extract:
            return

        sid = session_id or self._session_id
        if not sid:
            return

        self._turn_count += 1
        now = _now_iso()

        # User turn
        self._db.enqueue_turn(
            session_id=sid,
            turn_number=self._turn_count,
            role="user",
            content=user_content,
            timestamp=now,
            metadata_json=json.dumps({"author": turn_author} if turn_author else None, ensure_ascii=False),
        )
        self._turn_count += 1

        # Assistant turn
        self._db.enqueue_turn(
            session_id=sid,
            turn_number=self._turn_count,
            role="assistant",
            content=assistant_content,
            timestamp=now,
            metadata_json=None,
        )

        # Update session turn count
        self._db.increment_turn_count(sid)

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Mirror built-in MEMORY.md / USER.md writes into conversation memory."""
        if not self._db or not self._auto_extract or action not in {"add", "replace"} or not content:
            return

        sid = self._session_id or "memory-mirror"
        self._db.ensure_session(sid)

        label = "User Profile" if target == "user" else "Agent Memory"
        mirror_content = f"[{label} Write] {content}"

        self._turn_count += 1
        self._db.enqueue_turn(
            session_id=sid,
            turn_number=self._turn_count,
            role="tool",
            content=mirror_content,
            timestamp=_now_iso(),
            metadata_json=json.dumps({"source": "builtin_memory", "action": action, "target": target, **(metadata or {})}, ensure_ascii=False),
        )
        self._db.increment_turn_count(sid)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Extract insights from messages about to be compressed (v2 checkpoint API)."""
        if not self._db or not self._auto_extract or not messages:
            return ""

        # Skip messages already marked with _compressed_summary
        relevant = [
            m for m in messages
            if isinstance(m, dict)
            and m.get("role") in {"user", "assistant"}
            and isinstance(m.get("content"), str)
            and m.get("content", "").strip()
            and not m.get("_compressed_summary")
        ]
        if not relevant:
            return ""

        # Take last 10 relevant messages
        parts = [f"{m.get('role', '')}: {m.get('content', '')[:500]}" for m in relevant[-10:]]
        if not parts:
            return ""

        flush_content = "[Pre-compression context]\n" + "\n".join(parts)

        self._turn_count += 1
        self._db.enqueue_turn(
            session_id=self._session_id or "pre-compress",
            turn_number=self._turn_count,
            role="tool",
            content=flush_content,
            timestamp=_now_iso(),
            metadata_json=json.dumps({"source": "pre_compress", "message_count": len(parts)}, ensure_ascii=False),
        )
        if self._session_id:
            self._db.increment_turn_count(self._session_id)

        return ""  # Return empty; extracted content already persisted

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Flush pending writes at session boundary."""
        if self._db:
            self._db.flush(timeout=10.0)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Rebind to new session_id mid-process (/resume, /branch, /reset)."""
        if not self._db:
            return

        # Flush any pending writes for the old session
        self._db.flush(timeout=2.0)

        self._session_id = new_session_id
        self._turn_count = 0

        if not reset and not rewound:
            # Resume turn count
            self._turn_count = self._db.get_turn_count(self._session_id)

        self._db.ensure_session(self._session_id)

    def identity_signature(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "db_path": str(self._db._db_path) if self._db else "uninitialized",
        }

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return _ALL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if tool_name not in _TOOL_NAMES:
            return tool_error(f"Unknown tool: {tool_name}")

        if not self._db:
            return tool_error("Conversation memory not initialized")

        try:
            if tool_name == "conversation_memory_search":
                return self._tool_search(args)
            elif tool_name == "conversation_memory_recent":
                return self._tool_recent(args)
            elif tool_name == "conversation_memory_save":
                return self._tool_save(args)
            elif tool_name == "conversation_memory_status":
                return self._tool_status()
            elif tool_name == "conversation_memory_clear":
                return self._tool_clear(args)
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return tool_error(f"{tool_name} failed: {e}")

        return tool_error(f"Tool {tool_name} not handled")

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "").strip()
        if not query or len(query) < _MIN_QUERY_LEN:
            return json.dumps({"error": "Query too short (min 3 chars)"})

        session_id = args.get("session_id") or None
        limit = min(_coerce_int(args.get("limit"), 10), 50)
        role = args.get("role")

        results = self._db.search(query, session_id=session_id, limit=limit, role=role)
        if not results:
            return json.dumps({"results": [], "message": "No matches found"})

        out = []
        for r in results:
            out.append({
                "id": r.id,
                "session_id": r.session_id,
                "turn_number": r.turn_number,
                "role": r.role,
                "content": r.content[:1000],
                "timestamp": r.timestamp,
            })
        return json.dumps({"results": out, "count": len(out)})

    def _tool_recent(self, args: dict) -> str:
        session_id = args.get("session_id") or self._session_id
        if not session_id:
            return json.dumps({"error": "No session_id provided and no current session"})

        limit = min(_coerce_int(args.get("limit"), 20), 100)
        results = self._db.recent(session_id, limit=limit)

        out = []
        for r in results:
            out.append({
                "turn_number": r.turn_number,
                "role": r.role,
                "content": r.content[:1000],
                "timestamp": r.timestamp,
            })
        return json.dumps({"session_id": session_id, "turns": out, "count": len(out)})

    def _tool_save(self, args: dict) -> str:
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "Content required"})

        tags = args.get("tags", []) or []
        importance = float(args.get("importance", 0.7))

        sid = self._session_id or "explicit-save"
        self._db.ensure_session(sid)

        save_content = f"[Explicit Save] {content}"
        if tags:
            save_content += f" #tags: {', '.join(tags)}"

        self._turn_count += 1
        self._db.enqueue_turn(
            session_id=sid,
            turn_number=self._turn_count,
            role="tool",
            content=save_content,
            timestamp=_now_iso(),
            metadata_json=json.dumps({
                "source": "explicit_save",
                "tags": tags,
                "importance": importance,
            }, ensure_ascii=False),
        )
        self._db.increment_turn_count(sid)

        return json.dumps({"result": "Saved to conversation memory", "session_id": sid})

    def _tool_status(self) -> str:
        stats = self._db.get_stats()
        return json.dumps(stats, indent=2)

    def _tool_clear(self, args: dict) -> str:
        confirm = _coerce_bool(args.get("confirm"), False)
        if not confirm:
            return json.dumps({"error": "Deletion requires confirm=true"})

        session_id = args.get("session_id")
        if session_id:
            count = self._db.delete_session(session_id)
            return json.dumps({"result": f"Cleared session {session_id}", "deleted": count})
        else:
            count = self._db.delete_all_sessions()
            return json.dumps({"result": "Cleared ALL sessions", "deleted": count})

    def shutdown(self) -> None:
        if self._db:
            # Wait for prefetch thread
            if self._prefetch_thread and self._prefetch_thread.is_alive():
                self._prefetch_thread.join(timeout=3.0)
            self._db.flush(timeout=5.0)
            self._db.shutdown()
            self._db = None
        self._shutdown = True


# ──────────────────────────────────────────────────────────────────────────────
# Plugin Registration
# ──────────────────────────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Register Conversation Memory as a memory provider plugin."""
    ctx.register_memory_provider(ConversationMemoryProvider())