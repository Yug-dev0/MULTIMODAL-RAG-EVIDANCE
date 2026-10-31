from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.answer_synthesis import AnswerResult


def test_compare_endpoint_returns_both_systems_answers(auth_headers):
    client = TestClient(app)
    fake_multimodal = AnswerResult(answer="Redis, explained by the speaker over the architecture diagram.",
                                    citations=[], events_used=[])
    with (
        patch("app.api.compare.run_query", new=AsyncMock(return_value=fake_multimodal)),
        patch("app.api.compare.run_baseline_query", new=AsyncMock(return_value="Redis was mentioned.")),
    ):
        resp = client.get("/compare", params={"query": "What reduces database load?"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "diagram" in body["multimodal"]["answer"]
    assert body["baseline"]["answer"] == "Redis was mentioned."
