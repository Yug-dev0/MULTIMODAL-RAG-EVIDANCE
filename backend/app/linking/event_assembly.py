from app.storage.repository import persist_knowledge_event

EVENT_SCORE_THRESHOLD = 0.5
EVENT_TYPES = {
    "discussion", "decision", "proposal", "explanation", "diagram_explanation", "definition",
    "question", "answer", "comparison", "action_item", "observation", "reference",
}


def compute_event_score(temporal_overlap: float, entity_overlap: float,
                         semantic_similarity: float, relationship_confidence: float) -> float:
    return (0.35 * temporal_overlap + 0.30 * entity_overlap
            + 0.20 * semantic_similarity + 0.15 * relationship_confidence)


async def assemble_knowledge_event(session, *, event_type: str, title: str, summary: str,
                                    timestamp_start, timestamp_end, temporal_overlap: float,
                                    entity_overlap: float, semantic_similarity: float,
                                    relationship_confidence: float, member_units: list,
                                    entity_ids: list | None = None):
    assert event_type in EVENT_TYPES, f"invalid event_type: {event_type}"
    score = compute_event_score(temporal_overlap, entity_overlap, semantic_similarity, relationship_confidence)
    if score < EVENT_SCORE_THRESHOLD:
        return None
    event = await persist_knowledge_event(
        session, event_type=event_type, title=title, summary=summary,
        timestamp_start=timestamp_start, timestamp_end=timestamp_end,
        confidence=min(1.0, score), entities=entity_ids or [],
    )
    for unit in member_units:
        unit.event_id = event.id
    return event
