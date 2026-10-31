"""
Canonical structured store: every EvidenceUnit, in full, with provenance.
This is the source of truth — Chroma and Kuzu hold derived indexes
(vectors, graph edges) that point back to rows here by id.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import config
from src.schema.evidence_unit import EvidenceUnit
from src.schema.knowledge_event import KnowledgeEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_units (
    id TEXT PRIMARY KEY,
    modality TEXT NOT NULL,
    content TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_type TEXT NOT NULL,
    start_time REAL,
    end_time REAL,
    page_number INTEGER,
    speaker TEXT,
    confidence REAL NOT NULL,
    embedding_space TEXT,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL  -- full EvidenceUnit as JSON (entities, relationships,
                         -- bbox, provenance, cross_modal_links, etc.)
);
CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence_units(source_file);
CREATE INDEX IF NOT EXISTS idx_evidence_modality ON evidence_units(modality);
CREATE INDEX IF NOT EXISTS idx_evidence_time ON evidence_units(start_time, end_time);

CREATE TABLE IF NOT EXISTS source_files (
    source_file TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    unit_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS knowledge_events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    start_time REAL,
    end_time REAL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL  -- full KnowledgeEvent as JSON
);
"""


class SQLiteStore:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = str(db_path or config.SQLITE_PATH)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def upsert(self, units: list[EvidenceUnit]) -> None:
        rows = [
            (
                u.id, u.modality.value, u.content, u.source_file, u.source_type.value,
                u.start_time, u.end_time, u.page_number, u.speaker, u.confidence,
                u.embedding_space.value, u.created_at, u.model_dump_json(),
            )
            for u in units
        ]
        self._conn.executemany(
            """INSERT INTO evidence_units
               (id, modality, content, source_file, source_type, start_time, end_time,
                page_number, speaker, confidence, embedding_space, created_at, data)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 content=excluded.content, data=excluded.data, confidence=excluded.confidence,
                 embedding_space=excluded.embedding_space""",
            rows,
        )
        for source_file, source_type in {(u.source_file, u.source_type.value) for u in units}:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM evidence_units WHERE source_file = ?", (source_file,)
            ).fetchone()[0]
            self._conn.execute(
                """INSERT INTO source_files (source_file, source_type, ingested_at, unit_count)
                   VALUES (?, ?, datetime('now'), ?)
                   ON CONFLICT(source_file) DO UPDATE SET unit_count=excluded.unit_count""",
                (source_file, source_type, count),
            )
        self._conn.commit()

    def get(self, unit_id: str) -> EvidenceUnit | None:
        row = self._conn.execute("SELECT data FROM evidence_units WHERE id = ?", (unit_id,)).fetchone()
        return EvidenceUnit.model_validate_json(row["data"]) if row else None

    def get_many(self, unit_ids: list[str]) -> list[EvidenceUnit]:
        if not unit_ids:
            return []
        placeholders = ",".join("?" * len(unit_ids))
        rows = self._conn.execute(
            f"SELECT data FROM evidence_units WHERE id IN ({placeholders})", unit_ids
        ).fetchall()
        units = {u.id: u for u in (EvidenceUnit.model_validate_json(r["data"]) for r in rows)}
        return [units[i] for i in unit_ids if i in units]

    def all(self) -> list[EvidenceUnit]:
        rows = self._conn.execute("SELECT data FROM evidence_units").fetchall()
        return [EvidenceUnit.model_validate_json(r["data"]) for r in rows]

    def by_source(self, source_file: str) -> list[EvidenceUnit]:
        rows = self._conn.execute(
            "SELECT data FROM evidence_units WHERE source_file = ?", (source_file,)
        ).fetchall()
        return [EvidenceUnit.model_validate_json(r["data"]) for r in rows]

    def by_modality(self, modality: str) -> list[EvidenceUnit]:
        rows = self._conn.execute(
            "SELECT data FROM evidence_units WHERE modality = ?", (modality,)
        ).fetchall()
        return [EvidenceUnit.model_validate_json(r["data"]) for r in rows]

    def source_files(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM source_files").fetchall()

    # --- knowledge events --------------------------------------------------

    def upsert_events(self, events: list[KnowledgeEvent]) -> None:
        rows = [
            (e.id, e.title, e.start_time, e.end_time, e.confidence, e.created_at, e.model_dump_json())
            for e in events
        ]
        self._conn.executemany(
            """INSERT INTO knowledge_events (id, title, start_time, end_time, confidence, created_at, data)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET title=excluded.title, data=excluded.data""",
            rows,
        )
        self._conn.commit()

    def all_events(self) -> list[KnowledgeEvent]:
        rows = self._conn.execute("SELECT data FROM knowledge_events").fetchall()
        return [KnowledgeEvent.model_validate_json(r["data"]) for r in rows]

    def get_event(self, event_id: str) -> KnowledgeEvent | None:
        row = self._conn.execute("SELECT data FROM knowledge_events WHERE id = ?", (event_id,)).fetchone()
        return KnowledgeEvent.model_validate_json(row["data"]) if row else None

    def clear(self) -> None:
        self._conn.executescript(
            "DELETE FROM evidence_units; DELETE FROM source_files; DELETE FROM knowledge_events;"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
