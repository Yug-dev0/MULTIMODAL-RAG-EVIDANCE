import uuid
import pytest
from unittest.mock import patch, AsyncMock
from app.storage.models import KnowledgeEvent
from app.linking.event_assembly import compute_event_score, assemble_knowledge_event, EVENT_SCORE_THRESHOLD



def test_compute_event_score_matches_spec_formula():
    score = compute_event_score(temporal_overlap=0.8, entity_overlap=0.6,
                                 semantic_similarity=0.9, relationship_confidence=0.7)
    assert score == pytest.approx(0.35 * 0.8 + 0.30 * 0.6 + 0.20 * 0.9 + 0.15 * 0.7)


async def test_assemble_knowledge_event_creates_and_links_units():
    fake_event = KnowledgeEvent(id=uuid.uuid4(), event_type="diagram_explanation", title="t",
                                 summary="s", timestamp_start=0.0, timestamp_end=5.0, confidence=0.9,
                                 entities=[], metadata_json={})

    class FakeUnit:
        event_id = None

    unit = FakeUnit()
    with patch("app.linking.event_assembly.persist_knowledge_event", new=AsyncMock(return_value=fake_event)):
        result = await assemble_knowledge_event(
            None, event_type="diagram_explanation", title="t", summary="s",
            timestamp_start=0.0, timestamp_end=5.0, temporal_overlap=1.0, entity_overlap=1.0,
            semantic_similarity=1.0, relationship_confidence=1.0, member_units=[unit],
        )
    assert result is fake_event
    assert unit.event_id == fake_event.id


async def test_assemble_knowledge_event_returns_none_below_threshold():
    result = await assemble_knowledge_event(
        None, event_type="discussion", title="t", summary="s", timestamp_start=0.0, timestamp_end=1.0,
        temporal_overlap=0.1, entity_overlap=0.1, semantic_similarity=0.1, relationship_confidence=0.1,
        member_units=[],
    )
    assert result is None
    assert 0.1 * 0.35 + 0.1 * 0.30 + 0.1 * 0.20 + 0.1 * 0.15 < EVENT_SCORE_THRESHOLD
