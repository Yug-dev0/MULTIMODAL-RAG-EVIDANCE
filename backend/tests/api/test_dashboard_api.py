import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.storage.db import get_session
from app.storage.repository import (
    persist_evidence_unit, persist_entity, create_chat_session, link_source_to_session,
    persist_chat_message,
)
from app.ingestion.orchestrator import JOBS, JobStatus
from app.api.history import QUERY_LOG, record_query, QueryLogEntry


@pytest.mark.asyncio
@pytest.mark.integration
async def test_stats_summary_reflects_real_data(auth_headers):
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            await persist_evidence_unit(
                session, content_text="x", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                timestamp_start=0.0, timestamp_end=10.0, confidence=0.9,
            )
            await session.commit()

    client = TestClient(app)
    resp = client.get("/stats/summary", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["evidence_units"] >= 1
    assert body["sources"] >= 1
    assert "modality_distribution" in body
    assert "processing_complete_pct" in body


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evidence_endpoint_tags_each_row_with_its_chat_session(auth_headers):
    grouped_source = uuid.uuid4()
    ungrouped_source = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            chat_session = await create_chat_session(session, owner_uid="u1", title="Redis Architecture")
            await link_source_to_session(session, source_id=grouped_source, session_id=chat_session.id)
            await persist_evidence_unit(
                session, content_text="grouped", content_type="transcript", modality="speech",
                source_id=grouped_source, source_type="video", raw_locator="v", extraction_method="x",
                timestamp_start=430.0, timestamp_end=440.0, confidence=0.9,
            )
            await persist_evidence_unit(
                session, content_text="ungrouped", content_type="transcript", modality="speech",
                source_id=ungrouped_source, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            await session.commit()

    client = TestClient(app)
    resp = client.get("/evidence", headers=auth_headers)
    assert resp.status_code == 200
    by_text = {row["content_text"]: row for row in resp.json()}
    assert by_text["grouped"]["session_title"] == "Redis Architecture"
    assert by_text["grouped"]["session_id"] == str(chat_session.id)
    assert by_text["ungrouped"]["session_id"] is None
    assert by_text["ungrouped"]["session_title"] is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evidence_search_ranks_by_relevance_to_the_question(auth_headers):
    """The Evidence page's whole point is answering "what backs up what I asked".
    Without a query the endpoint is a recency listing, which is why an
    irrelevant row could outrank the one that actually matched."""
    source_id = uuid.uuid4()
    # Embeddings are stubbed so the ranking assertion tests our ordering, not
    # OpenAI's. The query vector is [1, 0, 0...]; "relevant" sits on top of it,
    # "irrelevant" is orthogonal -- and is written LAST so recency order would
    # put it first, making a pass impossible unless we genuinely rank.
    relevant_vec = [1.0] + [0.0] * 1535
    orthogonal_vec = [0.0, 1.0] + [0.0] * 1534

    async with get_session() as session:
        with patch("app.storage.repository.embed_texts", return_value=[relevant_vec]):
            await persist_evidence_unit(
                session, content_text="relevant passage text", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.5,
            )
        with patch("app.storage.repository.embed_texts", return_value=[orthogonal_vec]):
            await persist_evidence_unit(
                session, content_text="irrelevant passage text", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.99,
            )
        await session.commit()

    client = TestClient(app)
    with patch("app.api.dashboard.embed_texts", return_value=[relevant_vec]):
        resp = client.get("/evidence", params={"q": "the thing I asked about"}, headers=auth_headers)

    assert resp.status_code == 200
    body = resp.json()
    texts = [row["content_text"] for row in body]
    assert texts[0] == "relevant passage text", f"expected relevance ordering, got {texts}"
    # Relevance must be its own field -- the UI previously showed extraction
    # confidence, so the 0.99-confidence irrelevant row looked like a 99% match.
    assert body[0]["relevance"] > body[1]["relevance"]
    assert body[1]["confidence"] > body[0]["confidence"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evidence_without_query_keeps_listing_everything(auth_headers):
    """Browsing the corpus with no question asked must still work."""
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            await persist_evidence_unit(
                session, content_text="browsed", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            await session.commit()

    client = TestClient(app)
    resp = client.get("/evidence", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert [r["content_text"] for r in body] == ["browsed"]
    assert body[0]["relevance"] is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evidence_by_query_groups_each_answer_under_the_question_asked(auth_headers):
    """The page's job is "what backed up what I asked". Grouping by chat
    session scatters one question's evidence among everything else ingested
    in that session, so each question has to carry its own citations."""
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as db:
            chat_session = await create_chat_session(db, owner_uid="test-uid", title="Resume overview")
            await link_source_to_session(db, source_id=source_id, session_id=chat_session.id)
            cited = await persist_evidence_unit(
                db, content_text="Specializes in NLP and computer vision.", content_type="ocr_text",
                modality="document", source_id=source_id, source_type="pdf", extraction_method="x",
                raw_locator="r.pdf#page=1",
                page_number=1, confidence=0.9,
            )
            # Ingested in the same session but never cited -- this is exactly
            # what the session grouping used to mix in with the real answer.
            await persist_evidence_unit(
                db, content_text="Contact: someone@example.com", content_type="ocr_text",
                modality="document", source_id=source_id, source_type="pdf", extraction_method="x",
                raw_locator="r.pdf#page=2",
                page_number=2, confidence=0.99,
            )
            await persist_chat_message(db, session_id=chat_session.id, role="user",
                                       content="what are the specializations?")
            await persist_chat_message(
                db, session_id=chat_session.id, role="assistant", content="NLP and CV.",
                citations_json=[{"evidence_id": str(cited.id), "modality": "document"}],
            )
            await db.commit()

    client = TestClient(app)
    resp = client.get("/evidence/by-query", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    group = body[0]
    assert group["query"] == "what are the specializations?"
    assert group["session_title"] == "Resume overview"
    # Only the cited unit -- the uncited one must not ride along under a
    # heading that claims it answered the question.
    assert [e["content_text"] for e in group["evidence"]] == ["Specializes in NLP and computer vision."]
    assert group["evidence"][0]["raw_locator"] == "r.pdf#page=1"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evidence_by_query_hides_other_users_questions(auth_headers):
    """Questions are per-user; one account's history must not leak into another's."""
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as db:
            other = await create_chat_session(db, owner_uid="someone-else", title="Their chat")
            await persist_chat_message(db, session_id=other.id, role="user", content="their private question")
            await persist_chat_message(db, session_id=other.id, role="assistant", content="a", citations_json=[])
            await db.commit()

    client = TestClient(app)
    resp = client.get("/evidence/by-query", headers=auth_headers)
    assert resp.status_code == 200
    assert [g["query"] for g in resp.json()] == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_evidence_by_query_lists_newest_question_first(auth_headers):
    """Ordering is what makes the page usable after a long session -- the
    question just asked should be at the top, not buried under old ones."""
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as db:
            chat_session = await create_chat_session(db, owner_uid="test-uid", title="T")
            for text_ in ("first question", "second question"):
                await persist_chat_message(db, session_id=chat_session.id, role="user", content=text_)
                await persist_chat_message(db, session_id=chat_session.id, role="assistant",
                                           content="a", citations_json=[])
            await db.commit()

    client = TestClient(app)
    resp = client.get("/evidence/by-query", headers=auth_headers)
    assert [g["query"] for g in resp.json()] == ["second question", "first question"]


def test_jobs_endpoint_lists_submitted_jobs(auth_headers):
    JOBS["test-job-x"] = JobStatus(job_id="test-job-x", status="done", filename="clip.mp4", source_type="video")
    client = TestClient(app)
    resp = client.get("/jobs", headers=auth_headers)
    assert resp.status_code == 200
    ids = [j["job_id"] for j in resp.json()]
    assert "test-job-x" in ids


def test_queries_recent_reflects_logged_history(auth_headers):
    QUERY_LOG.clear()
    record_query(QueryLogEntry(query="What is Redis?", mode="query", citation_count=2, modality_count=2, baseline_citation_count=0))
    client = TestClient(app)
    resp = client.get("/queries/recent", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["query"] == "What is Redis?"


def test_eval_session_summary_is_honest_when_empty(auth_headers):
    QUERY_LOG.clear()
    client = TestClient(app)
    resp = client.get("/eval/session-summary", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["comparisons_run"] == 0
    assert body["avg_multimodal_citations"] == 0.0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_entities_top_endpoint_returns_ranked_real_entities(auth_headers):
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            redis = await persist_entity(session, canonical_name="Redis", entity_type="technology")
            unit = await persist_evidence_unit(
                session, content_text="x", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            unit.entities = [redis.id]
            await session.commit()

    client = TestClient(app)
    resp = client.get("/entities/top", headers=auth_headers)
    assert resp.status_code == 200
    names = [e["canonical_name"] for e in resp.json()]
    assert "Redis" in names
