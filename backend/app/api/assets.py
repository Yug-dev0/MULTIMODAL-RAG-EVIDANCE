import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.api.ingest import MEDIA_DIR

router = APIRouter()
_MEDIA_ROOT = os.path.realpath(MEDIA_DIR)


@router.get("/assets")
def get_asset(path: str):
    """Serves a thumbnail/frame image by its on-disk path, restricted to the
    media storage root — citations only ever reference files ingestion wrote
    there, so anything else is rejected rather than served."""
    resolved = os.path.realpath(path)
    if os.path.commonpath([resolved, _MEDIA_ROOT]) != _MEDIA_ROOT:
        raise HTTPException(status_code=403, detail="path outside media directory")
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(resolved)
