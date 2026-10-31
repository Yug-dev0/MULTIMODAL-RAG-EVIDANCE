import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.api.compare import router as compare_router
from app.api.graph import router as graph_router
from app.api.assets import router as assets_router
from app.api.dashboard import router as dashboard_router
from app.api.sessions import router as sessions_router
from app.api.auth import get_current_user

app = FastAPI(title="Multimodal RAG Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ingest_router, dependencies=[Depends(get_current_user)])
app.include_router(query_router, dependencies=[Depends(get_current_user)])
app.include_router(compare_router, dependencies=[Depends(get_current_user)])
app.include_router(graph_router, dependencies=[Depends(get_current_user)])
app.include_router(dashboard_router, dependencies=[Depends(get_current_user)])
app.include_router(sessions_router, dependencies=[Depends(get_current_user)])

# Deliberately NOT behind get_current_user: citation thumbnails are rendered
# with plain <img src> tags (see EvidenceCard.tsx), which cannot attach an
# Authorization header. answer_synthesis.py generates thumbnail URLs as
# /assets?path=..., which this router validates against the media root before
# serving. The /media static mount below is kept too (some earlier commits'
# pages may still link to it directly) — both are safe, unauthenticated,
# read-only, and independently scoped to MEDIA_STORAGE_DIR.
app.include_router(assets_router)

MEDIA_DIR = os.environ.get("MEDIA_STORAGE_DIR", "./data/media")
os.makedirs(MEDIA_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")


@app.get("/health")
def health():
    return {"status": "ok"}
