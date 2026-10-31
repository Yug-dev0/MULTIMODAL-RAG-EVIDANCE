import json
import uuid
from datetime import datetime, timedelta
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.storage.models import KnowledgeEvent, Relationship
from app.linking.contradiction import check_contradiction, detect_and_record_contradictions, ContradictionResult

pytestmark = pytest.mark.asyncio


async def test_check_contradiction_parses_structured_response():
    fake_message = MagicMock()
    fake_message.content = json.dumps({"contradicts": True, "confidence": 0.9, "explanation": "Later decision reverses the earlier proposal"})
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    with patch("app.linking.contradiction._call_contradiction_model", return_value=fake_completion):
        result = await check_contradiction("We'll use Redis.", "Actually, we decided not to use Redis.")
    assert isinstance(result, ContradictionResult)
    assert result.contradicts is True


async def test_detect_and_record_contradictions_persists_relationship_for_confirmed_pair():
    earlier = KnowledgeEvent(id=uuid.uuid4(), event_type="proposal", title="t1", summary="We'll use Redis.",
                              timestamp_start=0.0, timestamp_end=5.0, confidence=0.9, entities=[], metadata_json={})
    later = KnowledgeEvent(id=uuid.uuid4(), event_type="decision", title="t2", summary="We decided not to use Redis.",
                            timestamp_start=100.0, timestamp_end=105.0, confidence=0.9, entities=[], metadata_json={})
    fake_result = ContradictionResult(contradicts=True, confidence=0.85, explanation="reversal")
    fake_relationship = Relationship(id=uuid.uuid4(), source_type="knowledge_event", source_id=earlier.id,
                                      target_type="knowledge_event", target_id=later.id,
                                      relationship_type="contradicts", confidence=0.85, evidence_ids=[], metadata_json={})
    with (
        patch("app.linking.contradiction.check_contradiction", new=AsyncMock(return_value=fake_result)),
        patch("app.linking.contradiction.persist_relationship", new=AsyncMock(return_value=fake_relationship)) as mock_persist,
    ):
        created = await detect_and_record_contradictions(None, [later, earlier])
    assert len(created) == 1
    mock_persist.assert_awaited_once()
    _, kwargs = mock_persist.call_args
    assert kwargs["source_id"] == earlier.id and kwargs["target_id"] == later.id
    assert kwargs["metadata_json"]["superseded_by"] == str(later.id)


async def test_detect_and_record_contradictions_handles_null_timestamps():
    # Events from non-timestamped sources (images, PDFs) have timestamp_start=None;
    # sorting must fall back to created_at instead of comparing None < None.
    now = datetime.utcnow()
    earlier = KnowledgeEvent(id=uuid.uuid4(), event_type="observation", title="t1", summary="Redis is used.",
                              timestamp_start=None, timestamp_end=None, confidence=0.9, entities=[],
                              metadata_json={}, created_at=now)
    later = KnowledgeEvent(id=uuid.uuid4(), event_type="observation", title="t2", summary="Redis is not used.",
                            timestamp_start=None, timestamp_end=None, confidence=0.9, entities=[],
                            metadata_json={}, created_at=now + timedelta(seconds=1))
    with (
        patch("app.linking.contradiction.check_contradiction", new=AsyncMock(
            return_value=ContradictionResult(contradicts=False, confidence=0.1, explanation="unrelated"))),
        patch("app.linking.contradiction.persist_relationship", new=AsyncMock()),
    ):
        created = await detect_and_record_contradictions(None, [later, earlier])
    assert created == []
