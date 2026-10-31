import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.storage.repository import (
    persist_evidence_unit, persist_entity, corpus_stats, list_sources, top_entities,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_corpus_stats_reflects_real_rows():
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            await persist_evidence_unit(
                session, content_text="Redis reduces load", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v.mp4#t=0,5",
                extraction_method="whisper-1", timestamp_start=0.0, timestamp_end=5.0, confidence=0.9,
            )
            await persist_evidence_unit(
                session, content_text="REDIS DIAGRAM", content_type="ocr", modality="ocr",
                source_id=source_id, source_type="video", raw_locator="frame.jpg",
                extraction_method="tesseract", timestamp_start=2.0, timestamp_end=2.0, confidence=0.8,
            )
            await session.commit()
            stats = await corpus_stats(session)
    assert stats["evidence_units"] == 2
    assert stats["sources"] == 1
    assert stats["modality_distribution"] == {"speech": 1, "ocr": 1}
    assert stats["total_media_seconds"] == 5.0


async def test_list_sources_groups_by_source_id():
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            await persist_evidence_unit(
                session, content_text="x", content_type="document_text", modality="document",
                source_id=source_id, source_type="pdf", raw_locator="doc.pdf#page=1",
                extraction_method="pymupdf", page_number=1, confidence=0.98,
            )
            await session.commit()
            sources = await list_sources(session)
    assert len(sources) == 1
    assert sources[0]["source_type"] == "pdf"
    assert sources[0]["evidence_count"] == 1


async def test_top_entities_ranks_by_real_mention_count():
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            redis = await persist_entity(session, canonical_name="Redis", entity_type="technology")
            postgres = await persist_entity(session, canonical_name="PostgreSQL", entity_type="technology")
            u1 = await persist_evidence_unit(
                session, content_text="a", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            u2 = await persist_evidence_unit(
                session, content_text="b", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            u1.entities = [redis.id, postgres.id]
            u2.entities = [redis.id]
            await session.commit()
            ranked = await top_entities(session, limit=10)
    assert ranked[0]["canonical_name"] == "Redis"
    assert ranked[0]["mention_count"] == 2
    assert ranked[1]["canonical_name"] == "PostgreSQL"
    assert ranked[1]["mention_count"] == 1
