import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.storage.repository import persist_evidence_unit, persist_knowledge_event
from app.retrieval.vector_search import vector_search_evidence, vector_search_events

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_vector_search_skips_fragments_too_short_to_be_evidence():
    """ASR segmentation emits stubs like "AI." that embed close to any short
    question and then crowd out the real answer. A three-character unit can't
    support a claim no matter how well it scores."""
    source_id = uuid.uuid4()
    query_vec = [1.0] + [0.0] * 1535
    with patch("app.storage.repository.embed_texts") as mock_embed:
        async with get_session() as session:
            # The fragment sits exactly on the query vector, so it wins on pure
            # cosine distance -- it can only be excluded by the length rule.
            mock_embed.return_value = [query_vec]
            await persist_evidence_unit(
                session, content_text="AI.", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            mock_embed.return_value = [[0.9, 0.1] + [0.0] * 1534]
            substantive = await persist_evidence_unit(
                session, content_text="Specializes in NLP and computer vision research.",
                content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            await session.commit()
            results = await vector_search_evidence(session, query_vec, top_k=5)

    texts = [r[0].content_text for r in results]
    assert "AI." not in texts
    assert substantive.content_text in texts


async def test_vector_search_keeps_short_but_meaningful_evidence():
    """The cutoff must not eat genuinely terse evidence -- an OCR'd figure is
    short and still answers a question."""
    source_id = uuid.uuid4()
    query_vec = [1.0] + [0.0] * 1535
    with patch("app.storage.repository.embed_texts") as mock_embed:
        async with get_session() as session:
            mock_embed.return_value = [query_vec]
            terse = await persist_evidence_unit(
                session, content_text="Revenue: $4.2M", content_type="ocr_text", modality="ocr",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            await session.commit()
            results = await vector_search_evidence(session, query_vec, top_k=5)

    assert terse.content_text in [r[0].content_text for r in results]


async def test_vector_search_evidence_ranks_closest_first():
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts") as mock_embed:
        async with get_session() as session:
            mock_embed.return_value = [[1.0] + [0.0] * 1535]
            close = await persist_evidence_unit(
                session, content_text="close match passage", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            mock_embed.return_value = [[0.0, 1.0] + [0.0] * 1534]
            far = await persist_evidence_unit(
                session, content_text="far away passage", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            await session.commit()
            results = await vector_search_evidence(session, [1.0] + [0.0] * 1535, top_k=5)
    ids_in_order = [r[0].id for r in results]
    assert ids_in_order.index(close.id) < ids_in_order.index(far.id)


async def test_vector_search_events_returns_similarity_scores():
    with patch("app.storage.repository.embed_texts", return_value=[[1.0] + [0.0] * 1535]):
        async with get_session() as session:
            event = await persist_knowledge_event(
                session, event_type="discussion", title="t", summary="Redis caching",
                timestamp_start=0.0, timestamp_end=1.0, confidence=0.9,
            )
            await session.commit()
            results = await vector_search_events(session, [1.0] + [0.0] * 1535, top_k=5)
    assert any(r[0].id == event.id for r in results)


async def test_vector_search_evidence_respects_allowed_source_ids():
    # Regression guard for chat-session isolation: a query scoped to one
    # topic's sources must never surface evidence uploaded under another.
    allowed_source = uuid.uuid4()
    other_source = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[1.0] + [0.0] * 1535]):
        async with get_session() as session:
            in_scope = await persist_evidence_unit(
                session, content_text="in scope passage", content_type="transcript", modality="speech",
                source_id=allowed_source, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            out_of_scope = await persist_evidence_unit(
                session, content_text="out of scope passage", content_type="transcript", modality="speech",
                source_id=other_source, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            await session.commit()
            results = await vector_search_evidence(
                session, [1.0] + [0.0] * 1535, top_k=5, allowed_source_ids=[allowed_source],
            )
    ids = {r[0].id for r in results}
    assert in_scope.id in ids
    assert out_of_scope.id not in ids


async def test_vector_search_events_respects_allowed_source_ids_via_member_evidence():
    allowed_source = uuid.uuid4()
    other_source = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[1.0] + [0.0] * 1535]):
        async with get_session() as session:
            in_scope_event = await persist_knowledge_event(
                session, event_type="discussion", title="in scope", summary="s",
                timestamp_start=0.0, timestamp_end=1.0, confidence=0.9,
            )
            out_of_scope_event = await persist_knowledge_event(
                session, event_type="discussion", title="out of scope", summary="s",
                timestamp_start=0.0, timestamp_end=1.0, confidence=0.9,
            )
            await persist_evidence_unit(
                session, content_text="x", content_type="transcript", modality="speech",
                source_id=allowed_source, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9, event_id=in_scope_event.id,
            )
            await persist_evidence_unit(
                session, content_text="x", content_type="transcript", modality="speech",
                source_id=other_source, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9, event_id=out_of_scope_event.id,
            )
            await session.commit()
            results = await vector_search_events(
                session, [1.0] + [0.0] * 1535, top_k=5, allowed_source_ids=[allowed_source],
            )
    ids = {r[0].id for r in results}
    assert in_scope_event.id in ids
    assert out_of_scope_event.id not in ids
