"""
FastAPI app exposing ingestion status, hybrid query, multimodal-vs-baseline
comparison, and graph-browse endpoints over the pipeline's stores. Run with:

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from src.baseline.text_only_rag import run_baseline_query
from src.retrieval.query_engine import run_query
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

app = FastAPI(title="Multimodal RAG-Ready Pipeline")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sqlite = SQLiteStore()
_vectors = VectorStore()
_graph = GraphStore()

DATA_ROOT = (config.ROOT_DIR / "data").resolve()

# The self-contained single-file UI for this pipeline. Kept separate from
# `frontend/`, which is the other implementation's Vite/React app and whose
# index.html is a build entry point, not a servable page.
FRONTEND_DIR = Path(__file__).parent.parent / "frontend-standalone"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    units = _sqlite.all()
    by_modality: dict[str, int] = {}
    for u in units:
        by_modality[u.modality.value] = by_modality.get(u.modality.value, 0) + 1
    return {
        "sources": [dict(row) for row in _sqlite.source_files()],
        "unit_count": len(units),
        "by_modality": by_modality,
        "event_count": len(_sqlite.all_events()),
    }


@app.post("/query")
def query(request: QueryRequest):
    return run_query(request.query, sqlite=_sqlite, vectors=_vectors, graph=_graph, top_k=request.top_k)


@app.post("/compare")
def compare(request: QueryRequest):
    multimodal = run_query(request.query, sqlite=_sqlite, vectors=_vectors, graph=_graph, top_k=request.top_k)
    baseline = run_baseline_query(request.query, sqlite=_sqlite)
    return {"multimodal": multimodal, "baseline": baseline}


@app.get("/graph/{entity_name}")
def graph_neighborhood(entity_name: str, hops: int = 2):
    return {
        "entity": entity_name,
        "reachable_evidence": _graph.traverse_from_entity(entity_name, hops=hops),
        "related_entities": _graph.related_entities(entity_name),
    }


@app.get("/asset")
def asset(path: str):
    """Serves a raw extracted asset (frame/figure image) by absolute path,
    restricted to the project's data/ directory to prevent arbitrary
    filesystem reads."""
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(DATA_ROOT)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path outside data directory")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(resolved))
