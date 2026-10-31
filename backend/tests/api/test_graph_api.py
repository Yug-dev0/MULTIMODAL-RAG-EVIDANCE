from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app


def test_graph_endpoint_returns_nodes_and_links(auth_headers):
    client = TestClient(app)
    with patch("app.api.graph.fetch_graph_snapshot", new=AsyncMock(return_value={
        "nodes": [{"id": "1", "label": "Redis", "type": "entity"}],
        "links": [{"source": "1", "target": "2", "relationship_type": "explains"}],
    })):
        resp = client.get("/graph", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["nodes"][0]["label"] == "Redis"
