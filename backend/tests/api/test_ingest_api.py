import io
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app


def test_ingest_endpoint_returns_job_id_and_is_pollable(auth_headers):
    client = TestClient(app)
    with patch("app.api.ingest.run_ingestion_job", new=AsyncMock()):
        resp = client.post(
            "/ingest",
            files={"file": ("sample.mp4", io.BytesIO(b"fake bytes"), "video/mp4")},
            data={"source_type": "video"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        status_resp = client.get(f"/ingest/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] in ("pending", "running", "done")
