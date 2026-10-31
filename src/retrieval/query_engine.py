"""
Hybrid multimodal retrieval: embeds the query into both embedding spaces
(text and CLIP), searches Chroma's evidence/image/event collections,
expands from the query's own resolved entities via the Kuzu graph, fuses
everything into one confidence-weighted ranking, and returns a fully
provenanced evidence bundle plus a grounded natural-language answer.
"""

from __future__ import annotations

import logging

import config
from src.enrichment.embeddings import embed_text, embed_text_clip
from src.enrichment.llm_provider import get_llm_provider
from src.ingestion.entity_linking import resolve_entity_name_readonly
from src.schema.evidence_unit import EvidenceUnit
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _entities_for_query(query: str, graph: GraphStore) -> list[str]:
    provider = get_llm_provider()
    entities, _ = provider.extract_entities_relationships(query)
    resolved = []
    for entity in entities:
        name = resolve_entity_name_readonly(graph, entity.text)
        if name and name not in resolved:
            resolved.append(name)
    return resolved


def _to_bundle(unit: EvidenceUnit, index: int) -> dict:
    return {
        "index": index,
        "id": unit.id,
        "modality": unit.modality.value,
        "content": unit.content,
        "source_file": unit.source_file,
        "start_time": unit.start_time,
        "end_time": unit.end_time,
        "page_number": unit.page_number,
        "bbox": unit.bbox.model_dump() if unit.bbox else None,
        "confidence": unit.confidence,
        "speaker": unit.speaker,
        "asset_path": unit.provenance.raw_ref,
        "extractor": unit.provenance.extractor,
    }


def run_query(
    query: str, *, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore, top_k: int | None = None
) -> dict:
    top_k = top_k or config.FINAL_TOP_K
    query_text_vector = embed_text(query)
    query_clip_vector = embed_text_clip(query)

    text_hits = vectors.query_text(query_text_vector, top_k=config.RETRIEVAL_TOP_K_TEXT)
    image_hits = vectors.query_image(query_clip_vector, top_k=config.RETRIEVAL_TOP_K_IMAGE)
    event_hits = vectors.query_events(query_text_vector, top_k=3)

    vector_hits: dict[str, float] = {}
    for uid, sim in list(text_hits) + list(image_hits):
        vector_hits[uid] = max(vector_hits.get(uid, 0.0), sim)

    entity_names = _entities_for_query(query, graph)
    graph_hits: dict[str, float] = {}
    for name in entity_names:
        for uid, info in graph.traverse_from_entity(name, hops=config.GRAPH_TRAVERSAL_HOPS).items():
            graph_hits[uid] = max(graph_hits.get(uid, 0.0), info["confidence"])

    candidate_ids = set(vector_hits) | set(graph_hits)
    if not candidate_ids:
        return {
            "answer": "No relevant evidence was found in the ingested corpus for this query.",
            "evidence": [], "events": [], "query_entities": entity_names,
        }

    units = {u.id: u for u in sqlite.get_many(list(candidate_ids))}
    scored = []
    for uid in candidate_ids:
        unit = units.get(uid)
        if unit is None:
            continue
        score = (
            config.FUSION_WEIGHT_VECTOR * vector_hits.get(uid, 0.0)
            + config.FUSION_WEIGHT_GRAPH * graph_hits.get(uid, 0.0)
            + config.FUSION_WEIGHT_CONFIDENCE * unit.confidence
        )
        scored.append((unit, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    ranked_units = [u for u, _ in scored[:top_k]]
    ranked_ids = {u.id for u in ranked_units}

    events = []
    for event_id, sim in event_hits:
        event = sqlite.get_event(event_id)
        if event and (sim >= config.SEMANTIC_LINK_MIN_SIMILARITY or ranked_ids & set(event.evidence_ids)):
            events.append(event)

    evidence_bundles = [_to_bundle(u, i) for i, u in enumerate(ranked_units)]
    answer = get_llm_provider().synthesize_answer(query, evidence_bundles)

    return {
        "answer": answer,
        "evidence": evidence_bundles,
        "events": [e.model_dump() for e in events],
        "query_entities": entity_names,
    }
