"""
Standalone image ingestion — a screenshot, a photographed whiteboard, a
diagram exported on its own. Reuses the same region-aware OCR + LLM
captioning as the video adapter's keyframes, just without a timestamp.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.ingestion.persistence import enrich_and_persist
from src.ingestion.vision_common import caption_image, is_substantive_ocr, run_ocr_regions
from src.schema.evidence_unit import BoundingBox, EvidenceUnit, Modality, Provenance, SourceType
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_image(
    image_path: str, *, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore
) -> list[EvidenceUnit]:
    image_path = str(image_path)
    source_file = Path(image_path).name
    units: list[EvidenceUnit] = []

    caption, cap_conf = caption_image(image_path)
    image_unit = EvidenceUnit(
        modality=Modality.IMAGE,
        content=caption,
        source_file=source_file,
        source_type=SourceType.IMAGE,
        confidence=cap_conf,
        provenance=Provenance(extractor="llm-vision-caption", raw_ref=image_path),
    )
    enrich_and_persist(image_unit, sqlite=sqlite, vectors=vectors, graph=graph, image_path=image_path)
    units.append(image_unit)

    for region in run_ocr_regions(image_path):
        if not is_substantive_ocr(region["text"]):
            continue
        ocr_unit = EvidenceUnit(
            modality=Modality.OCR_TEXT,
            content=region["text"],
            source_file=source_file,
            source_type=SourceType.IMAGE,
            bbox=BoundingBox(**region["bbox"]),
            confidence=region["confidence"],
            provenance=Provenance(extractor="tesseract", raw_ref=image_path),
        )
        enrich_and_persist(ocr_unit, sqlite=sqlite, vectors=vectors, graph=graph)
        units.append(ocr_unit)

    logger.info("Ingested %d evidence units from %s", len(units), source_file)
    return units
