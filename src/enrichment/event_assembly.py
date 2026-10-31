"""
Event assembly: after cross-modal linking, clusters tightly-linked
EvidenceUnits (connected components over their temporal/semantic
CrossModalLinks — same_entity links are excluded here since they connect by
topic across the whole corpus, not by "this happened together") into
persisted KnowledgeEvent records: a title + summary from one LLM call over
the cluster, embedded and stored alongside evidence units so retrieval can
return a ready-made multimodal bundle instead of reassembling one from
scratch on every query.
"""

from __future__ import annotations

import logging

from src.enrichment.embeddings import embed_text
from src.enrichment.llm_provider import get_llm_provider
from src.schema.evidence_unit import EvidenceUnit
from src.schema.knowledge_event import KnowledgeEvent
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

CLUSTERING_LINK_TYPES = {"temporal_cooccurrence", "semantic_similarity"}
MIN_CLUSTER_CONFIDENCE = 0.3
MIN_CLUSTER_SIZE = 2
MAX_SNIPPETS_PER_EVENT = 12


def _connected_components(units: list[EvidenceUnit]) -> list[list[EvidenceUnit]]:
    by_id = {u.id: u for u in units}
    visited: set[str] = set()
    components: list[list[EvidenceUnit]] = []

    for start_unit in units:
        if start_unit.id in visited:
            continue
        stack = [start_unit.id]
        component_ids: list[str] = []
        while stack:
            uid = stack.pop()
            if uid in visited:
                continue
            visited.add(uid)
            component_ids.append(uid)
            unit = by_id.get(uid)
            if unit is None:
                continue
            for link in unit.cross_modal_links:
                if link.link_type not in CLUSTERING_LINK_TYPES or link.confidence < MIN_CLUSTER_CONFIDENCE:
                    continue
                if link.target_evidence_id in by_id and link.target_evidence_id not in visited:
                    stack.append(link.target_evidence_id)
        components.append([by_id[i] for i in component_ids])
    return components


def assemble_events(*, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore) -> list[KnowledgeEvent]:
    units = sqlite.all()
    components = _connected_components(units)
    provider = get_llm_provider()

    events: list[KnowledgeEvent] = []
    for component in components:
        if len(component) < MIN_CLUSTER_SIZE:
            continue
        modalities = sorted({u.modality.value for u in component})
        if len(modalities) < 2:
            continue  # only worth an "event" if it actually bridges modalities

        snippets = [u.content for u in component if u.content.strip()][:MAX_SNIPPETS_PER_EVENT]
        title, summary = provider.summarize_event(snippets)

        starts = [u.start_time for u in component if u.start_time is not None]
        ends = [u.end_time for u in component if u.end_time is not None]
        confidences = [u.confidence for u in component]

        event = KnowledgeEvent(
            title=title,
            summary=summary,
            evidence_ids=[u.id for u in component],
            modalities=modalities,
            source_files=sorted({u.source_file for u in component}),
            start_time=min(starts) if starts else None,
            end_time=max(ends) if ends else None,
            confidence=(sum(confidences) / len(confidences)) if confidences else 1.0,
        )
        events.append(event)

        sqlite.upsert_events([event])
        graph.upsert_event_node(event.id, event.title, event.start_time, event.end_time)
        for unit in component:
            graph.add_contains(event.id, unit.id)

        event_text = f"{title}. {summary}"
        vectors.add_event(
            event.id, embed_text(event_text),
            {
                "modalities": ",".join(modalities),
                "source_files": ",".join(event.source_files),
                "start_time": event.start_time,
                "end_time": event.end_time,
                "confidence": event.confidence,
            },
            document=event_text,
        )

    logger.info("Assembled %d multimodal knowledge events from %d connected components", len(events), len(components))
    return events
