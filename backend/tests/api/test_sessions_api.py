import io
from unittest.mock import patch, AsyncMock
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.answer_synthesis import AnswerResult, Citation

pytestmark = pytest.mark.integration


def test_new_session_then_list_returns_it():
    client = TestClient(app)
    created = client.post("/sessions").json()
    assert created["title"] == "New chat"
    assert created["message_count"] == 0
    assert created["source_count"] == 0

    sessions = client.get("/sessions").json()
    assert any(s["id"] == created["id"] for s in sessions)


def test_ingest_without_session_id_auto_creates_and_links_one():
    client = TestClient(app)
    with patch("app.api.ingest.run_ingestion_job", new=AsyncMock()):
        resp = client.post(
            "/ingest",
            files={"file": ("sample.mp4", io.BytesIO(b"fake bytes"), "video/mp4")},
            data={"source_type": "video"},
        )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    sessions = client.get("/sessions").json()
    assert any(s["id"] == session_id for s in sessions)


def test_ingest_with_existing_session_id_reuses_it():
    client = TestClient(app)
    created = client.post("/sessions").json()
    with patch("app.api.ingest.run_ingestion_job", new=AsyncMock()):
        resp = client.post(
            "/ingest",
            files={"file": ("sample.mp4", io.BytesIO(b"fake bytes"), "video/mp4")},
            data={"source_type": "video", "session_id": created["id"]},
        )
    assert resp.json()["session_id"] == created["id"]


def test_session_query_persists_messages_and_generates_title_on_first_exchange():
    client = TestClient(app)
    created = client.post("/sessions").json()
    fake_result = AnswerResult(
        answer="Redis was proposed to reduce database load.",
        citations=[Citation("id1", "speech", "video.mp4#t=1,2", "We propose Redis to reduce database load", 1.0, 2.0, None, None, 0.9)],
        events_used=["event1"],
    )
    with (
        patch("app.api.sessions.run_query", new=AsyncMock(return_value=fake_result)),
        patch("app.api.sessions._generate_session_title", return_value="Redis Caching Discussion"),
    ):
        resp = client.post(f"/sessions/{created['id']}/query", json={"query": "What reduces DB load?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == fake_result.answer
    assert body["session_title"] == "Redis Caching Discussion"

    messages = client.get(f"/sessions/{created['id']}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "What reduces DB load?"
    assert messages[1]["content"] == fake_result.answer
    assert len(messages[1]["citations"]) == 1

    sessions = client.get("/sessions").json()
    updated = next(s for s in sessions if s["id"] == created["id"])
    assert updated["title"] == "Redis Caching Discussion"
    assert updated["message_count"] == 2


def test_session_query_scopes_retrieval_to_that_sessions_sources():
    client = TestClient(app)
    created = client.post("/sessions").json()
    fake_result = AnswerResult(answer="a", citations=[], events_used=[])
    with (
        patch("app.api.sessions.run_query", new=AsyncMock(return_value=fake_result)) as mock_run_query,
        patch("app.api.sessions._generate_session_title", return_value="t"),
    ):
        client.post(f"/sessions/{created['id']}/query", json={"query": "q"})

    _, kwargs = mock_run_query.call_args
    assert kwargs["allowed_source_ids"] == []


def test_session_query_unknown_session_returns_404():
    client = TestClient(app)
    import uuid
    resp = client.post(f"/sessions/{uuid.uuid4()}/query", json={"query": "q"})
    assert resp.status_code == 404


def test_delete_session_removes_it_from_the_list():
    client = TestClient(app)
    created = client.post("/sessions").json()

    resp = client.delete(f"/sessions/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    sessions = client.get("/sessions").json()
    assert not any(s["id"] == created["id"] for s in sessions)


def test_delete_session_unknown_id_returns_404():
    client = TestClient(app)
    import uuid
    resp = client.delete(f"/sessions/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_delete_session_also_removes_its_messages_and_source_links():
    client = TestClient(app)
    created = client.post("/sessions").json()
    fake_result = AnswerResult(answer="a", citations=[], events_used=[])
    with (
        patch("app.api.sessions.run_query", new=AsyncMock(return_value=fake_result)),
        patch("app.api.sessions._generate_session_title", return_value="t"),
    ):
        client.post(f"/sessions/{created['id']}/query", json={"query": "q"})

    client.delete(f"/sessions/{created['id']}")

    # A stale reference to the deleted session's messages should 404, not
    # silently return orphaned rows.
    resp = client.get(f"/sessions/{created['id']}/messages")
    assert resp.status_code == 404
