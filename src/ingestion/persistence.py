"""
Shared "finish and persist" step used by every ingestion adapter: resolve
entity mentions to canonical graph nodes, embed the unit's content (text,
and CLIP image embedding when it carries a real image), and fan the result
out to all three stores (SQLite ledger, Chroma vector index, Kuzu graph) so
no adapter reimplements this by hand.
"""

from __future__ import annotations

import logging

from src.enrichment.embeddings import embed_image_clip, embed_text
from src.enrichment.llm_provider import get_llm_provider
from src.ingestion.entity_linking import resolve_entity_name
from src.schema.evidence_unit import EmbeddingSpace, Entity, EvidenceUnit, Modality, Relationship
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

IMAGE_LIKE_MODALITIES = {Modality.VIDEO_FRAME, Modality.IMAGE, Modality.PDF_IMAGE}


def _metadata(unit: EvidenceUnit) -> dict:
    return {
        "modality": unit.modality.value,
        "source_file": unit.source_file,
        "source_type": unit.source_type.value,
        "start_time": unit.start_time,
        "end_time": unit.end_time,
        "page_number": unit.page_number,
        "confidence": unit.confidence,
    }


def resolve_and_assign_entities(
    unit: EvidenceUnit, entities: list[Entity], relationships: list[Relationship], *, graph: GraphStore
) -> None:
    """Resolves each entity mention to its canonical graph name and assigns
    both lists onto `unit` in place — shared by the auto-extract path below
    and by callers that batch-extracted entities across many units upfront."""
    for entity in entities:
        entity.text = resolve_entity_name(graph, entity.text, entity.type)
    unit.entities = entities
    unit.relationships = relationships


def enrich_and_persist(
    unit: EvidenceUnit,
    *,
    sqlite: SQLiteStore,
    vectors: VectorStore,
    graph: GraphStore,
    image_path: str | None = None,
    extract_entities: bool = True,
) -> EvidenceUnit:
    """
    Mutates and returns `unit`: extracts + resolves entities/relationships
    from its content (unless the caller already assigned unit.entities via a
    batched extraction call and passes extract_entities=False), embeds it
    (text, and CLIP image embedding when `image_path` is given), and writes
    it to all three stores.
    """
    if extract_entities and len(unit.content.strip()) >= 3:
        provider = get_llm_provider()
        entities, relationships = provider.extract_entities_relationships(unit.content)
        resolve_and_assign_entities(unit, entities, relationships, graph=graph)

    embedded_text = False
    if unit.content.strip():
        text_vector = embed_text(unit.content)
        vectors.add_text(unit.id, text_vector, _metadata(unit), document=unit.content)
        embedded_text = True

    embedded_image = False
    if unit.modality in IMAGE_LIKE_MODALITIES and image_path:
        try:
            image_vector = embed_image_clip(image_path)
            vectors.add_image(unit.id, image_vector, _metadata(unit), document=unit.content)
            embedded_image = True
        except Exception as e:
            logger.warning("CLIP embedding failed for %s: %s", image_path, e)

    if embedded_text or embedded_image:
        unit.embedding_id = unit.id
        unit.embedding_space = EmbeddingSpace.IMAGE if embedded_image and not embedded_text else EmbeddingSpace.TEXT

    sqlite.upsert([unit])
    graph.upsert_evidence_node(unit)
    for entity in unit.entities:
        graph.upsert_entity_node(entity.text, entity.type)
        graph.add_mentions(unit.id, entity.text, entity.confidence)

    return unit
