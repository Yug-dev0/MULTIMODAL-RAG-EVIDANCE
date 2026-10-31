"""
PDF/document ingestion via PyMuPDF: text blocks (page number + bounding box
provenance) plus every embedded figure/image, which is run through the same
OCR + LLM captioning as a video keyframe. This is what lets a paper's
architecture diagram link up with the same diagram drawn live in a talk.

Entity extraction runs in two phases rather than once per unit: all raw
units are collected first, then their text is sent through the LLM provider
in a handful of batched calls (BATCH_SIZE items each). The free-tier API key
this project targets is request-rate-limited, not per-call-size-limited, so
this is the difference between minutes and hours for a document with
hundreds of small text blocks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import config
from src.enrichment.llm_provider import get_llm_provider
from src.ingestion.persistence import enrich_and_persist, resolve_and_assign_entities
from src.ingestion.vision_common import caption_image, is_substantive_ocr, run_ocr_regions
from src.schema.evidence_unit import BoundingBox, EvidenceUnit, Modality, Provenance, SourceType
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

TEXT_BLOCK_TYPE = 0
BATCH_SIZE = 150


def _batch_extract_and_persist(
    units: list[EvidenceUnit], *, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore,
    image_paths: dict[str, str] | None = None,
) -> None:
    """`image_paths` maps unit.id -> image file for units that also need a
    CLIP embedding (figures/frames) — their entities still come from the
    same batched call as everything else, only the caption itself required a
    separate (unavoidable) per-image vision call upstream."""
    image_paths = image_paths or {}
    provider = get_llm_provider()
    for start in range(0, len(units), BATCH_SIZE):
        batch = units[start : start + BATCH_SIZE]
        results = provider.extract_entities_relationships_batch([u.content for u in batch])
        for unit, (entities, relationships) in zip(batch, results):
            resolve_and_assign_entities(unit, entities, relationships, graph=graph)
            enrich_and_persist(
                unit, sqlite=sqlite, vectors=vectors, graph=graph, extract_entities=False,
                image_path=image_paths.get(unit.id),
            )


def ingest_pdf(
    pdf_path: str, *, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore
) -> list[EvidenceUnit]:
    import pymupdf

    pdf_path = str(pdf_path)
    source_file = Path(pdf_path).name
    out_dir = config.DATA_PROCESSED_DIR / Path(pdf_path).stem / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    text_units: list[EvidenceUnit] = []
    ocr_units: list[EvidenceUnit] = []
    figure_units: list[EvidenceUnit] = []
    figure_image_paths: dict[str, str] = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
            if block_type != TEXT_BLOCK_TYPE:
                continue
            text = text.strip()
            if len(text) < 3:
                continue
            text_units.append(EvidenceUnit(
                modality=Modality.PDF_TEXT,
                content=text,
                source_file=source_file,
                source_type=SourceType.PDF,
                page_number=page_number,
                bbox=BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                confidence=1.0,
                provenance=Provenance(extractor="pymupdf", raw_ref=f"{source_file}#page={page_number}"),
            ))

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                logger.warning("Could not extract image xref=%s on page %s: %s", xref, page_number, e)
                continue
            if base_image["width"] < 40 or base_image["height"] < 40:
                continue  # skip tiny/decorative images (icons, bullets)

            image_path = out_dir / f"page{page_number:03d}_img{img_index}.{base_image['ext']}"
            image_path.write_bytes(base_image["image"])

            bbox = None
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                bbox = BoundingBox(x=r.x0, y=r.y0, width=r.x1 - r.x0, height=r.y1 - r.y0)

            caption, cap_conf = caption_image(str(image_path))
            figure_unit = EvidenceUnit(
                modality=Modality.PDF_IMAGE,
                content=caption,
                source_file=source_file,
                source_type=SourceType.PDF,
                page_number=page_number,
                bbox=bbox,
                confidence=cap_conf,
                provenance=Provenance(extractor="llm-vision-caption", raw_ref=str(image_path)),
            )
            figure_units.append(figure_unit)
            figure_image_paths[figure_unit.id] = str(image_path)

            for region in run_ocr_regions(str(image_path)):
                if not is_substantive_ocr(region["text"]):
                    continue
                ocr_units.append(EvidenceUnit(
                    modality=Modality.OCR_TEXT,
                    content=region["text"],
                    source_file=source_file,
                    source_type=SourceType.PDF,
                    page_number=page_number,
                    bbox=BoundingBox(**region["bbox"]),
                    confidence=region["confidence"],
                    provenance=Provenance(extractor="tesseract", raw_ref=str(image_path)),
                ))

    doc.close()

    all_units = text_units + ocr_units + figure_units
    _batch_extract_and_persist(all_units, sqlite=sqlite, vectors=vectors, graph=graph, image_paths=figure_image_paths)

    logger.info("Ingested %d evidence units from %s", len(all_units), source_file)
    return all_units
