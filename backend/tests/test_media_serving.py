import os
from fastapi.testclient import TestClient


def test_media_directory_is_served_over_http(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_DIR", str(tmp_path))
    (tmp_path / "frame_0001.jpg").write_bytes(b"fake-jpeg-bytes")

    import importlib
    import app.main as main_module
    importlib.reload(main_module)

    client = TestClient(main_module.app)
    resp = client.get("/media/frame_0001.jpg")
    assert resp.status_code == 200
    assert resp.content == b"fake-jpeg-bytes"
