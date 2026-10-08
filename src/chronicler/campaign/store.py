"""SQLite persistence for campaigns, sessions, chunks and the campaign tracker.

One connection per Store, guarded by a lock so the web server, the pipeline
worker and background threads can share it safely.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE campaigns (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        notes_dir TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE sessions (
        id INTEGER PRIMARY KEY,
        campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
        title TEXT,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        status TEXT NOT NULL,              -- recording | stopped | finished
        dir TEXT NOT NULL,
        recap TEXT NOT NULL DEFAULT '',
        summary_json TEXT
    );
    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        audio_path TEXT NOT NULL,
        status TEXT NOT NULL,              -- queued | transcribing | analyzing | done | failed
        transcript_json TEXT,
        analysis_json TEXT,
        error TEXT,
        UNIQUE (session_id, idx)
    );
    CREATE TABLE entities (
        id INTEGER PRIMARY KEY,
        campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,                -- npc | place | item | faction | other
        aliases_json TEXT NOT NULL DEFAULT '[]',
        notes TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,              -- suggested | confirmed | dismissed
        first_session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE mentions (
        id INTEGER PRIMARY KEY,
        entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        note TEXT NOT NULL DEFAULT ''
    );
    CREATE TABLE threads (
        id INTEGER PRIMARY KEY,
        campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        state TEXT NOT NULL,               -- open | resolved
        status TEXT NOT NULL,              -- suggested | confirmed | dismissed
        first_session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE thread_events (
        id INTEGER PRIMARY KEY,
        thread_id INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
        session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,                -- opened | advanced | resolved
        note TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX idx_entities_campaign ON entities(campaign_id);
    CREATE INDEX idx_threads_campaign ON threads(campaign_id);
    CREATE INDEX idx_mentions_session ON mentions(session_id);
    CREATE INDEX idx_thread_events_session ON thread_events(session_id);
    """,
    # 2: entities that came from the campaign notes were tracked as new in the
    # session where they were first mentioned. The analysis of an entity's
    # first mention (before it existed in the tracker) flagged it `known` only
    # if it was in the notes, so use that flag to reclassify them as canon.
    """
    UPDATE entities SET first_session_id = NULL, status = 'confirmed'
    WHERE status = 'suggested' AND id IN (
        SELECT m.entity_id
        FROM mentions m
        JOIN chunks c ON c.id = m.chunk_id
        JOIN json_each(json_extract(c.analysis_json, '$.entities')) j
        JOIN entities e ON e.id = m.entity_id
        WHERE m.id = (SELECT MIN(m2.id) FROM mentions m2 WHERE m2.entity_id = m.entity_id)
          AND json_extract(j.value, '$.known') = 1
          AND lower(json_extract(j.value, '$.name')) = lower(e.name)
    );
    """,
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _norm(name: str) -> str:
    return " ".join(name.casefold().split())


@dataclass
class Campaign:
    id: int
    name: str
    notes_dir: str | None
    created_at: str


@dataclass
class Session:
    id: int
    campaign_id: int
    title: str | None
    started_at: str
    ended_at: str | None
    status: str
    dir: str
    recap: str
    summary_json: str | None

    @property
    def path(self) -> Path:
        return Path(self.dir)


@dataclass
class Chunk:
    id: int
    session_id: int
    idx: int
    start_s: float
    end_s: float
    audio_path: str
    status: str
    transcript_json: str | None
    analysis_json: str | None
    error: str | None

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return json.loads(self.transcript_json) if self.transcript_json else []

    @property
    def analysis(self) -> dict[str, Any] | None:
        return json.loads(self.analysis_json) if self.analysis_json else None


@dataclass
class Entity:
    id: int
    campaign_id: int
    name: str
    kind: str
    aliases: list[str]
    notes: str
    status: str
    first_session_id: int | None


@dataclass
class Thread:
    id: int
    campaign_id: int
    title: str
    state: str
    status: str
    first_session_id: int | None


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _q(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _migrate(self) -> None:
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
                self._conn.executescript(f"BEGIN; {script}; PRAGMA user_version = {i}; COMMIT;")

    # --- campaigns -----------------------------------------------------------

    def create_campaign(self, name: str, notes_dir: str | None = None) -> Campaign:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO campaigns (name, notes_dir, created_at) VALUES (?, ?, ?)",
                (name.strip(), notes_dir or None, now_iso()),
            )
            cid = cur.lastrowid
        assert cid is not None
        campaign = self.get_campaign(cid)
        assert campaign is not None
        return campaign

    def update_campaign(self, cid: int, *, name: str, notes_dir: str | None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE campaigns SET name = ?, notes_dir = ? WHERE id = ?",
                (name.strip(), notes_dir or None, cid),
            )

    def get_campaign(self, cid: int) -> Campaign | None:
        rows = self._q("SELECT * FROM campaigns WHERE id = ?", (cid,))
        return Campaign(**dict(rows[0])) if rows else None

    def list_campaigns(self) -> list[Campaign]:
        return [Campaign(**dict(r)) for r in self._q("SELECT * FROM campaigns ORDER BY id")]

    # --- sessions ------------------------------------------------------------

    def create_session(self, campaign_id: int, sessions_root: Path) -> Session:
        started = now_iso()
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO sessions (campaign_id, started_at, status, dir) VALUES (?, ?, ?, '')",
                (campaign_id, started, "recording"),
            )
            sid = cur.lastrowid
            assert sid is not None
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            session_dir = sessions_root / f"{stamp}_session-{sid}"
            c.execute("UPDATE sessions SET dir = ? WHERE id = ?", (str(session_dir), sid))
        session_dir.mkdir(parents=True, exist_ok=True)
        session = self.get_session(sid)
        assert session is not None
        return session

    def get_session(self, sid: int) -> Session | None:
        rows = self._q("SELECT * FROM sessions WHERE id = ?", (sid,))
        return Session(**dict(rows[0])) if rows else None

    def list_sessions(self, campaign_id: int) -> list[Session]:
        rows = self._q(
            "SELECT * FROM sessions WHERE campaign_id = ? ORDER BY started_at DESC, id DESC",
            (campaign_id,),
        )
        return [Session(**dict(r)) for r in rows]

    def set_session_status(self, sid: int, status: str) -> None:
        ended = now_iso() if status in ("stopped", "finished") else None
        with self.tx() as c:
            c.execute(
                "UPDATE sessions SET status = ?, ended_at = COALESCE(?, ended_at) WHERE id = ?",
                (status, ended, sid),
            )

    def set_session_recap(self, sid: int, recap: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE sessions SET recap = ? WHERE id = ?", (recap, sid))

    def set_session_summary(self, sid: int, title: str, summary_json: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE sessions SET title = ?, summary_json = ? WHERE id = ?",
                (title, summary_json, sid),
            )

    def mark_interrupted_sessions(self) -> list[int]:
        """Sessions left in `recording` after a crash become `stopped`."""
        rows = self._q("SELECT id FROM sessions WHERE status = 'recording'")
        ids = [r["id"] for r in rows]
        for sid in ids:
            self.set_session_status(sid, "stopped")
        return ids

    # --- chunks --------------------------------------------------------------

    def add_chunk(
        self, session_id: int, idx: int, start_s: float, end_s: float, audio_path: str
    ) -> Chunk:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO chunks (session_id, idx, start_s, end_s, audio_path, status) "
                "VALUES (?, ?, ?, ?, ?, 'queued')",
                (session_id, idx, start_s, end_s, audio_path),
            )
            chunk_id = cur.lastrowid
        assert chunk_id is not None
        chunk = self.get_chunk(chunk_id)
        assert chunk is not None
        return chunk

    def get_chunk(self, chunk_id: int) -> Chunk | None:
        rows = self._q("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        return Chunk(**dict(rows[0])) if rows else None

    def list_chunks(self, session_id: int) -> list[Chunk]:
        rows = self._q("SELECT * FROM chunks WHERE session_id = ? ORDER BY idx", (session_id,))
        return [Chunk(**dict(r)) for r in rows]

    def pending_chunks(self) -> list[Chunk]:
        """Chunks that were queued or mid-flight when the app last stopped."""
        rows = self._q(
            "SELECT * FROM chunks WHERE status IN ('queued', 'transcribing', 'analyzing') "
            "ORDER BY session_id, idx"
        )
        return [Chunk(**dict(r)) for r in rows]

    def set_chunk_status(self, chunk_id: int, status: str, error: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE chunks SET status = ?, error = ? WHERE id = ?", (status, error, chunk_id)
            )

    def set_chunk_transcript(self, chunk_id: int, segments: list[dict[str, Any]]) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE chunks SET transcript_json = ? WHERE id = ?",
                (json.dumps(segments, ensure_ascii=False), chunk_id),
            )

    def set_chunk_analysis(self, chunk_id: int, analysis: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE chunks SET analysis_json = ? WHERE id = ?",
                (json.dumps(analysis, ensure_ascii=False), chunk_id),
            )

    # --- entities ------------------------------------------------------------

    @staticmethod
    def _entity(row: sqlite3.Row | dict[str, Any]) -> Entity:
        d = dict(row)
        d["aliases"] = json.loads(d.pop("aliases_json"))
        d.pop("created_at", None)
        return Entity(**d)

    def list_entities(self, campaign_id: int, statuses: tuple[str, ...] = ()) -> list[Entity]:
        sql = "SELECT * FROM entities WHERE campaign_id = ?"
        params: tuple[Any, ...] = (campaign_id,)
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            params += statuses
        sql += " ORDER BY kind, name COLLATE NOCASE"
        return [self._entity(r) for r in self._q(sql, params)]

    def get_entity(self, entity_id: int) -> Entity | None:
        rows = self._q("SELECT * FROM entities WHERE id = ?", (entity_id,))
        return self._entity(rows[0]) if rows else None

    def find_entity(self, campaign_id: int, name: str) -> Entity | None:
        """Case/whitespace-insensitive lookup by name or alias. Ignores dismissed entities."""
        target = _norm(name)
        for e in self.list_entities(campaign_id, ("suggested", "confirmed")):
            if _norm(e.name) == target or any(_norm(a) == target for a in e.aliases):
                return e
        return None

    def add_entity(
        self,
        campaign_id: int,
        name: str,
        kind: str,
        *,
        status: str = "suggested",
        notes: str = "",
        aliases: list[str] | None = None,
        first_session_id: int | None = None,
    ) -> Entity:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO entities (campaign_id, name, kind, aliases_json, notes, status, "
                "first_session_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    name.strip(),
                    kind,
                    json.dumps(aliases or [], ensure_ascii=False),
                    notes,
                    status,
                    first_session_id,
                    now_iso(),
                ),
            )
            eid = cur.lastrowid
        assert eid is not None
        entity = self.get_entity(eid)
        assert entity is not None
        return entity

    def update_entity(
        self,
        entity_id: int,
        *,
        name: str | None = None,
        kind: str | None = None,
        notes: str | None = None,
        status: str | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        for col, val in (("name", name), ("kind", kind), ("notes", notes), ("status", status)):
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(val.strip() if col == "name" else val)
        if aliases is not None:
            sets.append("aliases_json = ?")
            params.append(json.dumps(aliases, ensure_ascii=False))
        if not sets:
            return
        with self.tx() as c:
            c.execute(f"UPDATE entities SET {', '.join(sets)} WHERE id = ?", (*params, entity_id))

    def merge_entities(self, source_id: int, target_id: int) -> None:
        """Fold `source` into `target`: its name and aliases become target aliases."""
        if source_id == target_id:
            return
        source = self.get_entity(source_id)
        target = self.get_entity(target_id)
        if source is None or target is None:
            raise ValueError("unknown entity")
        aliases = list(target.aliases)
        for alias in [source.name, *source.aliases]:
            if _norm(alias) != _norm(target.name) and all(
                _norm(alias) != _norm(a) for a in aliases
            ):
                aliases.append(alias)
        notes = target.notes
        if source.notes and source.notes not in notes:
            notes = f"{notes}\n{source.notes}".strip()
        with self.tx() as c:
            c.execute(
                "UPDATE mentions SET entity_id = ? WHERE entity_id = ?", (target_id, source_id)
            )
            c.execute(
                "UPDATE entities SET aliases_json = ?, notes = ? WHERE id = ?",
                (json.dumps(aliases, ensure_ascii=False), notes, target_id),
            )
            c.execute("DELETE FROM entities WHERE id = ?", (source_id,))

    def add_mention(self, entity_id: int, session_id: int, chunk_id: int, note: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO mentions (entity_id, session_id, chunk_id, note) VALUES (?, ?, ?, ?)",
                (entity_id, session_id, chunk_id, note),
            )

    def session_entities(self, session_id: int) -> list[tuple[Entity, list[str], bool]]:
        """Entities mentioned in a session: (entity, notes from this session, is_new)."""
        rows = self._q(
            "SELECT e.*, GROUP_CONCAT(m.note, char(31)) AS mention_notes "
            "FROM mentions m JOIN entities e ON e.id = m.entity_id "
            "WHERE m.session_id = ? AND e.status != 'dismissed' "
            "GROUP BY e.id ORDER BY MIN(m.id)",
            (session_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            notes = [n for n in (d.pop("mention_notes") or "").split("\x1f") if n]
            entity = self._entity(d)
            out.append((entity, notes, entity.first_session_id == session_id))
        return out

    def clear_chunk_effects(self, chunk_id: int) -> None:
        """Remove tracker rows produced by a chunk, so re-analysis doesn't duplicate them."""
        with self.tx() as c:
            c.execute("DELETE FROM mentions WHERE chunk_id = ?", (chunk_id,))
            c.execute("DELETE FROM thread_events WHERE chunk_id = ?", (chunk_id,))

    # --- threads -------------------------------------------------------------

    def list_threads(
        self,
        campaign_id: int,
        statuses: tuple[str, ...] = (),
        states: tuple[str, ...] = (),
    ) -> list[Thread]:
        sql = "SELECT * FROM threads WHERE campaign_id = ?"
        params: tuple[Any, ...] = (campaign_id,)
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            params += statuses
        if states:
            sql += f" AND state IN ({','.join('?' * len(states))})"
            params += states
        sql += " ORDER BY id"
        rows = self._q(sql, params)
        return [Thread(**{k: v for k, v in dict(r).items() if k != "created_at"}) for r in rows]

    def get_thread(self, thread_id: int) -> Thread | None:
        rows = self._q("SELECT * FROM threads WHERE id = ?", (thread_id,))
        if not rows:
            return None
        return Thread(**{k: v for k, v in dict(rows[0]).items() if k != "created_at"})

    def find_thread(self, campaign_id: int, title: str) -> Thread | None:
        target = _norm(title)
        for t in self.list_threads(campaign_id, ("suggested", "confirmed")):
            if _norm(t.title) == target:
                return t
        return None

    def add_thread(
        self,
        campaign_id: int,
        title: str,
        *,
        status: str = "suggested",
        first_session_id: int | None = None,
    ) -> Thread:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO threads (campaign_id, title, state, status, first_session_id, "
                "created_at) VALUES (?, ?, 'open', ?, ?, ?)",
                (campaign_id, title.strip(), status, first_session_id, now_iso()),
            )
            tid = cur.lastrowid
        assert tid is not None
        thread = self.get_thread(tid)
        assert thread is not None
        return thread

    def update_thread(
        self,
        thread_id: int,
        *,
        title: str | None = None,
        state: str | None = None,
        status: str | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        for col, val in (("title", title), ("state", state), ("status", status)):
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(val.strip() if col == "title" else val)
        if not sets:
            return
        with self.tx() as c:
            c.execute(f"UPDATE threads SET {', '.join(sets)} WHERE id = ?", (*params, thread_id))

    def add_thread_event(
        self, thread_id: int, session_id: int, chunk_id: int, kind: str, note: str
    ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO thread_events (thread_id, session_id, chunk_id, kind, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, session_id, chunk_id, kind, note),
            )

    def session_threads(self, session_id: int) -> list[tuple[Thread, list[tuple[str, str]]]]:
        """Threads touched in a session with their (kind, note) events, in order."""
        rows = self._q(
            "SELECT t.id AS tid, te.kind, te.note FROM thread_events te "
            "JOIN threads t ON t.id = te.thread_id "
            "WHERE te.session_id = ? AND t.status != 'dismissed' ORDER BY te.id",
            (session_id,),
        )
        grouped: dict[int, list[tuple[str, str]]] = {}
        for r in rows:
            grouped.setdefault(r["tid"], []).append((r["kind"], r["note"]))
        out = []
        for tid, events in grouped.items():
            thread = self.get_thread(tid)
            if thread:
                out.append((thread, events))
        return out
