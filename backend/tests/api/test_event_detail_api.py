import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.storage.db import get_session
from app.storage.repository import persist_evidence_unit, persist_entity, persist_knowledge_event, persist_relationship


@pytest.mark.asyncio
@pytest.mark.integration
async def test_event_detail_returns_real_evidence_relationships_and_entities():
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            redis = await persist_entity(session, canonical_name="Redis", entity_type="technology")
            event = await persist_knowledge_event(
                session, event_type="diagram_explanation", title="Redis caching proposed",
                summary="Rahul proposed Redis as a caching layer.", timestamp_start=12.4, timestamp_end=13.0,
                confidence=0.9, entities=[redis.id],
            )
            speech = await persist_evidence_unit(
                session, content_text="We should use Redis.", content_type="transcript", modality="speech",
                source_id=uuid.uuid4(), source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9, event_id=event.id,
            )
            frame = await persist_evidence_unit(
                session, content_text="REDIS ARCHITECTURE", content_type="ocr", modality="ocr",
                source_id=uuid.uuid4(), source_type="video", raw_locator="f.jpg", extraction_method="x",
                confidence=0.85, event_id=event.id,
            )
            await persist_relationship(
                session, source_type="evidence", source_id=speech.id, target_type="evidence",
                target_id=frame.id, relationship_type="explains", confidence=0.93,
            )
            await session.commit()

    client = TestClient(app)
    resp = client.get(f"/events/{event.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Redis caching proposed"
    assert len(body["evidence"]) == 2
    assert len(body["relationships"]) == 1
    assert body["relationships"][0]["relationship_type"] == "explains"
    assert body["entities"][0]["canonical_name"] == "Redis"


def test_event_detail_404_for_unknown_id():
    client = TestClient(app)
    resp = client.get(f"/events/{uuid.uuid4()}")
    assert resp.status_code == 404
