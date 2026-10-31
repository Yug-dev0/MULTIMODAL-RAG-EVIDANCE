import asyncio
from collections import defaultdict
from app.storage.repository import get_evidence_units_by_source, persist_relationship
from app.linking.segmentation import segment_into_windows
from app.linking.candidate_scoring import score_candidates
from app.linking.relationship_inference import infer_relationship
from app.linking.entity_resolution import resolve_entity
from app.linking.event_assembly import assemble_knowledge_event
from app.linking.summarization import summarize_event
from app.linking.contradiction import detect_and_record_contradictions
from app.linking.graph_sync import sync_to_graph
from app.extraction.ner import extract_entities


async def _noop_stage(_stage: str) -> None:
    return None


async def _resolve_unit_entities(session, unit, mentions, resolved_entities: dict) -> None:
    # Sequential on purpose: resolve_entity reads/writes the shared session,
    # which cannot be used concurrently. The expensive part (NER extraction)
    # is already gathered for every unit before this runs — see link_source.
    for mention in mentions:
        entity = await resolve_entity(session, mention.text, mention.entity_type)
        resolved_entities[entity.id] = entity
        if entity.id not in unit.entities:
            unit.entities = unit.entities + [entity.id]


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


async def _process_window(window, speech_units, candidate_units) -> dict:
    """
    All the LLM-bound work for one window (relationship inference per
    candidate, event summarization) with no session access, so this can run
    fully concurrently across every window in link_source — the session-bound
    steps (persisting relationship edges, assembling the knowledge event)
    happen afterward, in order.
    """
    window_units = [u for u in speech_units if u.id in window.evidence_unit_ids]
    window_candidates = [
        u for u in candidate_units
        if u.timestamp_start is not None and window.start - 15 <= u.timestamp_start <= window.end + 15
    ]
    ranked = score_candidates(window_units, window_candidates, top_k=3) if window_candidates else []

    window_text = " ".join(u.content_text for u in window_units)
    inferences = await asyncio.gather(
        *(infer_relationship(window_text, candidate.content_text, candidate.modality) for candidate, _ in ranked)
    )
    accepted = [(candidate, inference) for (candidate, _), inference in zip(ranked, inferences) if inference.related]

    speech_entities = {eid for u in window_units for eid in u.entities}
    candidate_entities = {eid for c, _ in accepted for eid in c.entities}
    entity_overlap = _jaccard(speech_entities, candidate_entities) if accepted else 1.0
    semantic_similarity = (sum(r.confidence for _, r in accepted) / len(accepted)) if accepted else 1.0

    evidence_texts = [u.content_text for u in window_units] + [c.content_text for c, _ in accepted]
    summary = await summarize_event(evidence_texts)

    return {
        "window": window, "window_units": window_units, "accepted": accepted,
        "entity_overlap": entity_overlap, "semantic_similarity": semantic_similarity,
        "relationship_confidence": semantic_similarity, "summary": summary,
        "entity_ids": list(speech_entities | candidate_entities),
    }


async def link_source(session, driver, source_id, on_stage=None) -> dict:
    on_stage = on_stage or _noop_stage
    units = await get_evidence_units_by_source(session, source_id)
    speech_units = [u for u in units if u.modality == "speech"]
    candidate_units = [u for u in units if u.modality in ("ocr", "image", "document")]

    # NER extraction is a pure API call (no session access) — gather it for
    # every unit up front instead of one at a time, then resolve/persist each
    # unit's mentions sequentially against the shared session.
    await on_stage("understand")
    resolved_entities: dict = {}
    all_mentions = await asyncio.gather(*(extract_entities(u.content_text) for u in units))
    for unit, mentions in zip(units, all_mentions):
        await _resolve_unit_entities(session, unit, mentions, resolved_entities)

    scene_timestamps = sorted({u.timestamp_start for u in candidate_units if u.timestamp_start is not None})
    windows = segment_into_windows(speech_units, scene_timestamps)

    await on_stage("link")
    window_results = await asyncio.gather(
        *(_process_window(window, speech_units, candidate_units) for window in windows)
    )

    events = []
    relationships = []
    for r in window_results:
        for wu in r["window_units"]:
            for candidate, inference in r["accepted"]:
                # Persist the actual cross-modal edge -- previously these
                # inferences only influenced event clustering and were
                # discarded, so /graph and the Relationships stat never
                # reflected the real explains/depicts/corresponds_to links.
                rel = await persist_relationship(
                    session, source_type="evidence", source_id=wu.id,
                    target_type="evidence", target_id=candidate.id,
                    relationship_type=inference.relationship_type, confidence=inference.confidence,
                    timestamp_start=wu.timestamp_start, timestamp_end=candidate.timestamp_start,
                    evidence_ids=[wu.id, candidate.id],
                    metadata_json={"evidence": inference.evidence},
                )
                relationships.append(rel)

        event = await assemble_knowledge_event(
            session, event_type=r["summary"].event_type, title=r["summary"].title, summary=r["summary"].summary,
            timestamp_start=r["window"].start, timestamp_end=r["window"].end, temporal_overlap=1.0,
            entity_overlap=r["entity_overlap"], semantic_similarity=r["semantic_similarity"],
            relationship_confidence=r["relationship_confidence"],
            member_units=r["window_units"] + [c for c, _ in r["accepted"]],
            entity_ids=r["entity_ids"],
        )
        if event:
            events.append(event)

    leftover = [u for u in units if u.event_id is None]
    leftover_summaries = await asyncio.gather(*(summarize_event([u.content_text]) for u in leftover))
    for unit, summary in zip(leftover, leftover_summaries):
        event = await assemble_knowledge_event(
            session, event_type=summary.event_type, title=summary.title, summary=summary.summary,
            timestamp_start=unit.timestamp_start, timestamp_end=unit.timestamp_end,
            temporal_overlap=1.0, entity_overlap=1.0, semantic_similarity=1.0, relationship_confidence=1.0,
            member_units=[unit], entity_ids=list(unit.entities),
        )
        if event:
            events.append(event)

    entity_to_events = defaultdict(list)
    for event in events:
        for entity_id in event.entities:
            entity_to_events[entity_id].append(event)

    contradictions = []
    for entity_id, entity_events in entity_to_events.items():
        if len(entity_events) > 1:
            contradictions.extend(await detect_and_record_contradictions(session, entity_events))
    relationships.extend(contradictions)

    await on_stage("index")
    await sync_to_graph(
        driver, entities=list(resolved_entities.values()),
        evidence_units=units, events=events, relationships=relationships,
    )

    return {
        "windows": len(windows), "events_created": len(events),
        "relationships_created": len(relationships), "contradictions_found": len(contradictions),
    }
