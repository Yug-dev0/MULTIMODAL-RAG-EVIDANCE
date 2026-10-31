"""
Standalone audio ingestion (no visual channel) — a voice note, a recorded
call, an addendum clip. Shares the same transcription helper as the video
adapter so confidence scoring is identical either way.
"""

from __future__ import annotations

import logging
from pathlib import Path

import config
from src.ingestion.audio_common import transcribe
from src.ingestion.persistence import enrich_and_persist
from src.schema.evidence_unit import EvidenceUnit, Modality, Provenance, SourceType
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_audio(
    audio_path: str, *, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore
) -> list[EvidenceUnit]:
    audio_path = str(audio_path)
    source_file = Path(audio_path).name

    logger.info("Transcribing %s", audio_path)
    segments = transcribe(audio_path)

    units: list[EvidenceUnit] = []
    for seg in segments:
        unit = EvidenceUnit(
            modality=Modality.TRANSCRIPT_SEGMENT,
            content=seg["text"],
            source_file=source_file,
            source_type=SourceType.AUDIO,
            start_time=seg["start"],
            end_time=seg["end"],
            confidence=seg["confidence"],
            provenance=Provenance(extractor="faster-whisper", model=config.WHISPER_MODEL_SIZE, raw_ref=audio_path),
        )
        enrich_and_persist(unit, sqlite=sqlite, vectors=vectors, graph=graph)
        units.append(unit)

    logger.info("Ingested %d evidence units from %s", len(units), source_file)
    return units
