import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.storage.repository import persist_evidence_unit, persist_entity, get_evidence_units_by_source

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_persist_evidence_unit_embeds_and_saves():
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            unit = await persist_evidence_unit(
                session, content_text="Redis reduces DB load", content_type="transcript",
                modality="speech", source_id=source_id, source_type="video",
                raw_locator="video.mp4#t=1,2", extraction_method="whisper-1",
                timestamp_start=1.0, timestamp_end=2.0, confidence=0.9,
            )
            await session.commit()
            assert unit.id is not None
            assert unit.embedding is not None
            fetched = await get_evidence_units_by_source(session, source_id)
            assert len(fetched) == 1
            assert fetched[0].content_text == "Redis reduces DB load"


async def test_persist_entity_defaults():
    with patch("app.storage.repository.embed_texts", return_value=[[0.2] * 1536]):
        async with get_session() as session:
            entity = await persist_entity(session, canonical_name="Redis", entity_type="technology")
            await session.commit()
            assert entity.aliases == []
