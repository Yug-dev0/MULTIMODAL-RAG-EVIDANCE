from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.answer_synthesis import AnswerResult, Citation


def test_query_endpoint_returns_answer_and_citations(auth_headers):
    client = TestClient(app)
    fake_result = AnswerResult(
        answer="Redis was proposed to reduce database load.",
        citations=[Citation("id1", "speech", "video.mp4#t=1,2", "We propose Redis to reduce database load", 1.0, 2.0, None, None, 0.9)],
        events_used=["event1"],
    )
    with patch("app.api.query.run_query", new=AsyncMock(return_value=fake_result)):
        resp = client.post("/query", json={"query": "What architecture reduces database load?"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == fake_result.answer
    assert len(body["citations"]) == 1
