# Multimodal RAG-Ready Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working multimodal ingestion → structured cross-modal knowledge graph → hybrid retrieval pipeline (plus a naive text-only baseline for comparison) that can answer queries whose evidence spans video, audio, images, and PDFs, with full provenance.

**Architecture:** Four first-class objects (EvidenceUnit, KnowledgeEvent, Entity, Relationship) stored in Postgres+pgvector and mirrored into Neo4j as a property graph. Ingestion normalizes every modality into EvidenceUnits; a linking stage builds semantic segments, scores candidate cross-modal pairs cheaply before any LLM call, infers typed relationships, resolves entities, assembles KnowledgeEvents via a deterministic weighted score, and tracks contradictions. Retrieval fuses vector search, event search, and graph traversal, resolves contradictions by recency, and synthesizes a cited answer. A naive fixed-chunk baseline runs the same queries for comparison.

**Tech Stack:** Python 3.11 / FastAPI, OpenAI (Whisper, GPT-4o, GPT-4o-mini, text-embedding-3-small), pyannote.audio, Tesseract, PyMuPDF, ffmpeg + PySceneDetect, Postgres 16 + pgvector, Neo4j 5, React + Vite + TS + Tailwind + 3d-force-graph, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-22-multimodal-rag-pipeline-design.md`

## Global Constraints

- Python 3.11, FastAPI, SQLAlchemy 2.x (async), Alembic for migrations.
- `relationship_type` values are restricted to exactly: `explains, supports, corresponds_to, mentions, depicts, contradicts, follows, references, unrelated` — never freeform.
- `event_type` values are restricted to exactly: `discussion, decision, proposal, explanation, diagram_explanation, definition, question, answer, comparison, action_item, observation, reference`.
- All LLM structured-extraction calls (NER, relationship inference, event summarization) use OpenAI structured outputs (`response_format={"type": "json_schema", ...}` with `strict: true`) — never freeform JSON parsing.
- Every EvidenceUnit must carry `extraction_method`, `extracted_at`, and a `raw_locator` sufficient to recover the exact source span (timestamp range, or page+bbox).
- No fixed-size chunking anywhere in the multimodal path — only in `backend/app/baseline/` (the naive comparison system), where fixed-size chunking is the deliberate point of contrast.
- Secrets (`OPENAI_API_KEY`, DB URLs, `HF_TOKEN`) come from `.env`, never hardcoded; `.env.example` documents every key.
- Do not push to any git remote at any point in this plan — local commits only.

---

## Phase 0 — Scaffolding

### Task 1: Repo scaffolding, Docker Compose, env config

**Files:**
- Create: `backend/app/__init__.py`, `backend/app/main.py`, `backend/requirements.txt`, `backend/Dockerfile`
- Create: `frontend/.gitkeep` (frontend scaffolded properly in Phase 6)
- Create: `docker-compose.yml`, `.env.example`, `.gitignore`, `README.md`
- Test: `backend/tests/test_health.py`

**Interfaces:**
- Produces: `GET /health` on the FastAPI app returning `{"status": "ok"}` — later tasks mount routers onto this same `app` instance from `backend/app/main.py`.

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.pyc
.env
node_modules/
dist/
.venv/
data/media/
*.db
```

- [ ] **Step 2: Create `.env.example`**

```
OPENAI_API_KEY=
HF_TOKEN=
POSTGRES_URL=postgresql+asyncpg://ragdb:ragdb@localhost:5432/ragdb
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=ragdb_neo4j
MEDIA_STORAGE_DIR=./data/media
```

- [ ] **Step 3: Write `backend/requirements.txt`**

```
fastapi==0.115.*
uvicorn[standard]==0.32.*
sqlalchemy[asyncio]==2.0.*
asyncpg==0.30.*
pgvector==0.3.*
alembic==1.14.*
neo4j==5.27.*
openai==1.57.*
pydantic==2.10.*
pydantic-settings==2.6.*
python-multipart==0.0.*
ffmpeg-python==0.2.*
scenedetect==0.6.*
pytesseract==0.3.*
Pillow==11.*
pymupdf==1.25.*
pyannote.audio==3.3.*
numpy==2.2.*
pytest==8.3.*
pytest-asyncio==0.24.*
httpx==0.28.*
```

- [ ] **Step 4: Write the failing test**

```python
# backend/tests/test_health.py
from fastapi.testclient import TestClient
from app.main import app

def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_health.py -v`
Expected: FAIL (ModuleNotFoundError: app.main, since it doesn't exist yet)

- [ ] **Step 6: Write minimal `backend/app/main.py`**

```python
from fastapi import FastAPI

app = FastAPI(title="Multimodal RAG Pipeline")

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_health.py -v`
Expected: PASS

- [ ] **Step 8: Write `docker-compose.yml`**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ragdb
      POSTGRES_PASSWORD: ragdb
      POSTGRES_DB: ragdb
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/ragdb_neo4j
    ports: ["7474:7474", "7687:7687"]
    volumes: ["neo4jdata:/data"]
  backend:
    build: ./backend
    env_file: .env
    ports: ["8000:8000"]
    volumes: ["./backend:/app", "./data/media:/app/data/media"]
    depends_on: [postgres, neo4j]
  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    depends_on: [backend]
volumes:
  pgdata:
  neo4jdata:
```

- [ ] **Step 9: Write `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg tesseract-ocr libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 10: Commit**

```bash
git add backend/ docker-compose.yml .env.example .gitignore
git commit -m "chore: scaffold backend, docker-compose, env config"
```

### Task 2: Postgres models for the four core objects + baseline_chunks

**Files:**
- Create: `backend/app/storage/__init__.py`, `backend/app/storage/db.py`, `backend/app/storage/models.py`
- Create: `backend/alembic.ini`, `backend/migrations/env.py`, `backend/migrations/versions/0001_initial.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Produces: SQLAlchemy async engine/session via `backend/app/storage/db.py::get_session() -> AsyncSession` (async context manager) and `Base` (declarative base). ORM classes `EvidenceUnit`, `Entity`, `KnowledgeEvent`, `Relationship`, `BaselineChunk` in `backend/app/storage/models.py`, fields exactly as named in the spec's schema blocks (snake_case). `Relationship.relationship_type` and `KnowledgeEvent.event_type` are `sa.Enum` bound to Python `enum.Enum` classes `RelationshipType` and `EventType` (also exported from `models.py`) with the exact members from Global Constraints.
- Consumes: nothing (foundational).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_models.py
import pytest
from app.storage.models import EvidenceUnit, Entity, KnowledgeEvent, Relationship, RelationshipType, EventType

def test_relationship_type_enum_values():
    assert {e.value for e in RelationshipType} == {
        "explains", "supports", "corresponds_to", "mentions", "depicts",
        "contradicts", "follows", "references", "unrelated",
    }

def test_event_type_enum_values():
    assert {e.value for e in EventType} == {
        "discussion", "decision", "proposal", "explanation",
        "diagram_explanation", "definition", "question", "answer",
        "comparison", "action_item", "observation", "reference",
    }

def test_evidence_unit_has_required_columns():
    cols = {c.name for c in EvidenceUnit.__table__.columns}
    assert {
        "id", "content_text", "content_type", "modality", "source_id",
        "source_type", "raw_locator", "extraction_method", "extracted_at",
        "timestamp_start", "timestamp_end", "page_number", "bbox",
        "embedding", "confidence", "confidence_factors", "event_id",
        "metadata_json", "searchable_text", "language",
    }.issubset(cols)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_models.py -v`
Expected: FAIL (ModuleNotFoundError: app.storage.models)

- [ ] **Step 3: Implement `backend/app/storage/db.py`**

```python
import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

DATABASE_URL = os.environ.get("POSTGRES_URL", "postgresql+asyncpg://ragdb:ragdb@localhost:5432/ragdb")
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

@asynccontextmanager
async def get_session():
    async with SessionLocal() as session:
        yield session
```

- [ ] **Step 4: Implement `backend/app/storage/models.py`**

```python
import enum
import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.storage.db import Base

EMBEDDING_DIM = 1536  # text-embedding-3-small

class RelationshipType(str, enum.Enum):
    explains = "explains"
    supports = "supports"
    corresponds_to = "corresponds_to"
    mentions = "mentions"
    depicts = "depicts"
    contradicts = "contradicts"
    follows = "follows"
    references = "references"
    unrelated = "unrelated"

class EventType(str, enum.Enum):
    discussion = "discussion"
    decision = "decision"
    proposal = "proposal"
    explanation = "explanation"
    diagram_explanation = "diagram_explanation"
    definition = "definition"
    question = "question"
    answer = "answer"
    comparison = "comparison"
    action_item = "action_item"
    observation = "observation"
    reference = "reference"

def _uuid_col():
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

class KnowledgeEvent(Base):
    __tablename__ = "knowledge_events"
    id: Mapped[uuid.UUID] = _uuid_col()
    event_type: Mapped[EventType] = mapped_column(SAEnum(EventType, name="event_type_enum"))
    title: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(String)
    timestamp_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    entities: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_units: Mapped[list["EvidenceUnit"]] = relationship(back_populates="event")

class Entity(Base):
    __tablename__ = "entities"
    id: Mapped[uuid.UUID] = _uuid_col()
    canonical_name: Mapped[str] = mapped_column(String, index=True)
    entity_type: Mapped[str] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

class EvidenceUnit(Base):
    __tablename__ = "evidence_units"
    id: Mapped[uuid.UUID] = _uuid_col()
    content_text: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(String)
    modality: Mapped[str] = mapped_column(String)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    source_type: Mapped[str] = mapped_column(String)
    raw_locator: Mapped[str] = mapped_column(String)
    extraction_method: Mapped[str] = mapped_column(String)
    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    timestamp_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    entities: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    confidence: Mapped[float] = mapped_column(Float)
    confidence_factors: Mapped[dict] = mapped_column(JSON, default=dict)
    event_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("knowledge_events.id"), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    searchable_text: Mapped[str] = mapped_column(String)
    language: Mapped[str] = mapped_column(String, default="en")
    event: Mapped[KnowledgeEvent | None] = relationship(back_populates="evidence_units")

class Relationship(Base):
    __tablename__ = "relationships"
    id: Mapped[uuid.UUID] = _uuid_col()
    source_type: Mapped[str] = mapped_column(String)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    target_type: Mapped[str] = mapped_column(String)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    relationship_type: Mapped[RelationshipType] = mapped_column(SAEnum(RelationshipType, name="relationship_type_enum"))
    confidence: Mapped[float] = mapped_column(Float)
    timestamp_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

class BaselineChunk(Base):
    __tablename__ = "baseline_chunks"
    id: Mapped[uuid.UUID] = _uuid_col()
    content_text: Mapped[str] = mapped_column(String)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    source_type: Mapped[str] = mapped_column(String)
    chunk_index: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
```

Note: `metadata` is a reserved attribute name on SQLAlchemy declarative models, so every table uses `metadata_json` as the column/attribute name for the spec's `metadata` field — this rename is intentional and every later task must use `metadata_json`, not `metadata`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 6: Initialize Alembic and write the initial migration**

Run: `cd backend && alembic init migrations`, then edit `migrations/env.py` to import `Base` from `app.storage.db` and `app.storage.models` (so `target_metadata = Base.metadata`), then:

Run: `alembic revision --autogenerate -m "initial schema"` (after Postgres is up via `docker compose up -d postgres`) followed by `alembic upgrade head`.
Expected: migration file created under `migrations/versions/`, tables exist in the running Postgres (verify with `docker compose exec postgres psql -U ragdb -c '\dt'`).

- [ ] **Step 7: Commit**

```bash
git add backend/app/storage backend/alembic.ini backend/migrations backend/tests/test_models.py
git commit -m "feat: add Postgres models for EvidenceUnit/Entity/KnowledgeEvent/Relationship"
```

### Task 3: Neo4j client + graph sync helpers

**Files:**
- Create: `backend/app/storage/graph.py`
- Test: `backend/tests/test_graph.py` (requires a running Neo4j — mark with `@pytest.mark.integration`)

**Interfaces:**
- Produces: `backend/app/storage/graph.py::get_driver() -> neo4j.AsyncDriver`, `upsert_entity_node(tx, entity: dict) -> None`, `upsert_evidence_node(tx, evidence: dict) -> None`, `upsert_event_node(tx, event: dict) -> None`, `upsert_relationship_edge(tx, rel: dict) -> None` where each `dict` has exactly the field names from the corresponding SQLAlchemy model in Task 2 (so callers can pass `model_to_dict(orm_instance)`). Also `setup_constraints(driver) -> None` creating uniqueness constraints on `Entity.id`, `EvidenceUnit.id`, `KnowledgeEvent.id`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_graph.py
import pytest
from app.storage.graph import get_driver, setup_constraints, upsert_entity_node, upsert_evidence_node, upsert_relationship_edge

pytestmark = pytest.mark.asyncio

async def test_entity_roundtrip():
    driver = get_driver()
    await setup_constraints(driver)
    async with driver.session() as session:
        await session.execute_write(upsert_entity_node, {
            "id": "11111111-1111-1111-1111-111111111111",
            "canonical_name": "Redis", "entity_type": "technology",
            "description": None, "aliases": ["redis-cache"], "metadata_json": {},
        })
        result = await session.run(
            "MATCH (e:Entity {id: $id}) RETURN e.canonical_name AS name",
            id="11111111-1111-1111-1111-111111111111",
        )
        record = await result.single()
        assert record["name"] == "Redis"
    await driver.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_graph.py -v`
Expected: FAIL (ModuleNotFoundError: app.storage.graph)

- [ ] **Step 3: Implement `backend/app/storage/graph.py`**

```python
import os
from neo4j import AsyncGraphDatabase

def get_driver():
    return AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "ragdb_neo4j")),
    )

async def setup_constraints(driver):
    async with driver.session() as session:
        for label in ("Entity", "EvidenceUnit", "KnowledgeEvent"):
            await session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

async def upsert_entity_node(tx, entity: dict):
    await tx.run(
        "MERGE (e:Entity {id: $id}) SET e.canonical_name=$canonical_name, "
        "e.entity_type=$entity_type, e.description=$description, e.aliases=$aliases",
        **{k: entity.get(k) for k in ("id", "canonical_name", "entity_type", "description", "aliases")},
    )

async def upsert_evidence_node(tx, evidence: dict):
    await tx.run(
        "MERGE (v:EvidenceUnit {id: $id}) SET v.content_text=$content_text, v.modality=$modality, "
        "v.source_id=$source_id, v.timestamp_start=$timestamp_start, v.timestamp_end=$timestamp_end, "
        "v.confidence=$confidence",
        **{k: evidence.get(k) for k in (
            "id", "content_text", "modality", "source_id", "timestamp_start", "timestamp_end", "confidence"
        )},
    )

async def upsert_event_node(tx, event: dict):
    await tx.run(
        "MERGE (ev:KnowledgeEvent {id: $id}) SET ev.event_type=$event_type, ev.title=$title, "
        "ev.summary=$summary, ev.timestamp_start=$timestamp_start, ev.timestamp_end=$timestamp_end, "
        "ev.confidence=$confidence",
        **{k: event.get(k) for k in (
            "id", "event_type", "title", "summary", "timestamp_start", "timestamp_end", "confidence"
        )},
    )

async def upsert_relationship_edge(tx, rel: dict):
    await tx.run(
        "MATCH (a {id: $source_id}), (b {id: $target_id}) "
        "MERGE (a)-[r:RELATES {relationship_type: $relationship_type}]->(b) "
        "SET r.confidence=$confidence, r.metadata_json=$metadata_json, r.id=$id",
        source_id=str(rel["source_id"]), target_id=str(rel["target_id"]),
        relationship_type=rel["relationship_type"], confidence=rel["confidence"],
        metadata_json=str(rel.get("metadata_json", {})), id=str(rel["id"]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose up -d neo4j` then `cd backend && python -m pytest tests/test_graph.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/storage/graph.py backend/tests/test_graph.py
git commit -m "feat: add Neo4j client and node/edge upsert helpers"
```

---

## Phase 1 — Extraction primitives

All extraction modules wrap an external tool/API behind a small, pure-Python-testable interface. Tasks that call OpenAI/pyannote mock the network call in tests and add one `@pytest.mark.integration` test that hits the real API, so the suite runs offline by default.

### Task 4: Video utilities — audio extraction + scene-based keyframes

**Files:**
- Create: `backend/app/extraction/video.py`
- Test: `backend/tests/extraction/test_video.py`
- Fixture: `backend/tests/fixtures/sample.mp4` (short, few-second clip checked into the repo for tests — user or Claude supplies one real short clip; see Task placement note below)

**Interfaces:**
- Produces: `extract_audio(video_path: str, out_dir: str) -> str` (returns path to extracted `.wav`), `extract_keyframes(video_path: str, out_dir: str) -> list[Keyframe]` where `Keyframe = namedtuple("Keyframe", ["timestamp", "image_path"])`, both defined in `backend/app/extraction/video.py`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/extraction/test_video.py
import os
from app.extraction.video import extract_audio, extract_keyframes

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample.mp4")

def test_extract_audio_produces_wav(tmp_path):
    wav_path = extract_audio(FIXTURE, str(tmp_path))
    assert os.path.exists(wav_path)
    assert wav_path.endswith(".wav")

def test_extract_keyframes_produces_timestamped_images(tmp_path):
    frames = extract_keyframes(FIXTURE, str(tmp_path))
    assert len(frames) >= 1
    for f in frames:
        assert os.path.exists(f.image_path)
        assert isinstance(f.timestamp, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/extraction/test_video.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/extraction/video.py`**

```python
import os
import subprocess
from collections import namedtuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

Keyframe = namedtuple("Keyframe", ["timestamp", "image_path"])

def extract_audio(video_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", wav_path],
        check=True, capture_output=True,
    )
    return wav_path

def extract_keyframes(video_path: str, out_dir: str) -> list[Keyframe]:
    os.makedirs(out_dir, exist_ok=True)
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    frames = []
    if not scenes:
        scenes = [(video.base_timecode, video.duration)]
    for i, (start, _end) in enumerate(scenes):
        ts = start.get_seconds()
        image_path = os.path.join(out_dir, f"frame_{i:04d}.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path, "-frames:v", "1", image_path],
            check=True, capture_output=True,
        )
        frames.append(Keyframe(timestamp=ts, image_path=image_path))
    return frames
```

- [ ] **Step 4: Add a short real test fixture and run**

Place any short (5-15s) real `.mp4` at `backend/tests/fixtures/sample.mp4` (screen recording or phone clip is fine — it only needs to exist and be decodable).

Run: `cd backend && python -m pytest tests/extraction/test_video.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/extraction/video.py backend/tests/extraction/test_video.py backend/tests/fixtures/sample.mp4
git commit -m "feat: add ffmpeg audio extraction and scene-based keyframe extraction"
```

### Task 5: ASR module — Whisper transcription with derived confidence

**Files:**
- Create: `backend/app/extraction/asr.py`
- Test: `backend/tests/extraction/test_asr.py`

**Interfaces:**
- Produces: `TranscriptSegment` dataclass (`start: float, end: float, text: str, avg_logprob: float, no_speech_prob: float, confidence: float, speaker: str | None`) and `transcribe(audio_path: str) -> list[TranscriptSegment]` in `backend/app/extraction/asr.py`. `confidence` is derived as `max(0.0, min(1.0, 1.0 + avg_logprob)) * (1.0 - no_speech_prob)`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/extraction/test_asr.py
from unittest.mock import patch, MagicMock
from app.extraction.asr import transcribe, TranscriptSegment

def test_transcribe_maps_segments_and_confidence():
    fake_response = MagicMock()
    fake_response.segments = [
        MagicMock(start=0.0, end=2.5, text="Hello world", avg_logprob=-0.1, no_speech_prob=0.02),
    ]
    with patch("app.extraction.asr._call_whisper", return_value=fake_response):
        segments = transcribe("dummy/path.wav")
    assert len(segments) == 1
    seg = segments[0]
    assert isinstance(seg, TranscriptSegment)
    assert seg.text == "Hello world"
    assert seg.start == 0.0 and seg.end == 2.5
    assert 0.0 <= seg.confidence <= 1.0
    assert seg.speaker is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/extraction/test_asr.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/extraction/asr.py`**

```python
import os
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    confidence: float
    speaker: str | None = None

def _call_whisper(audio_path: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(audio_path, "rb") as f:
        return client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

def _confidence(avg_logprob: float, no_speech_prob: float) -> float:
    base = max(0.0, min(1.0, 1.0 + avg_logprob))
    return base * (1.0 - no_speech_prob)

def transcribe(audio_path: str) -> list[TranscriptSegment]:
    response = _call_whisper(audio_path)
    return [
        TranscriptSegment(
            start=s.start, end=s.end, text=s.text,
            avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
            confidence=_confidence(s.avg_logprob, s.no_speech_prob),
        )
        for s in response.segments
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/extraction/test_asr.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/extraction/asr.py backend/tests/extraction/test_asr.py
git commit -m "feat: add Whisper ASR wrapper with derived confidence"
```

### Task 6: Diarization module + speaker-merge

**Files:**
- Create: `backend/app/extraction/diarization.py`
- Test: `backend/tests/extraction/test_diarization.py`

**Interfaces:**
- Produces: `SpeakerTurn` dataclass (`start: float, end: float, speaker: str`), `diarize(audio_path: str) -> list[SpeakerTurn]`, and `merge_speakers(segments: list[TranscriptSegment], turns: list[SpeakerTurn]) -> list[TranscriptSegment]` (returns new `TranscriptSegment` list with `.speaker` set to the turn whose interval has the largest overlap with the segment; `None` if no overlap) in `backend/app/extraction/diarization.py`.
- Consumes: `TranscriptSegment` from `app.extraction.asr` (Task 5).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/extraction/test_diarization.py
from unittest.mock import patch, MagicMock
from app.extraction.asr import TranscriptSegment
from app.extraction.diarization import diarize, merge_speakers, SpeakerTurn

def test_diarize_returns_speaker_turns():
    fake_annotation = MagicMock()
    fake_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=3.0), None, "SPEAKER_00"),
        (MagicMock(start=3.0, end=6.0), None, "SPEAKER_01"),
    ]
    with patch("app.extraction.diarization._run_pipeline", return_value=fake_annotation):
        turns = diarize("dummy/path.wav")
    assert turns == [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(3.0, 6.0, "SPEAKER_01")]

def test_merge_speakers_assigns_by_max_overlap():
    segments = [TranscriptSegment(0.5, 2.5, "hi", -0.05, 0.01, 0.9)]
    turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(3.0, 6.0, "SPEAKER_01")]
    merged = merge_speakers(segments, turns)
    assert merged[0].speaker == "SPEAKER_00"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/extraction/test_diarization.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/extraction/diarization.py`**

```python
import os
from dataclasses import dataclass, replace
from app.extraction.asr import TranscriptSegment

@dataclass(eq=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str

_pipeline = None

def _run_pipeline(audio_path: str):
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=os.environ.get("HF_TOKEN"),
        )
    return _pipeline(audio_path)

def diarize(audio_path: str) -> list[SpeakerTurn]:
    annotation = _run_pipeline(audio_path)
    return [
        SpeakerTurn(start=segment.start, end=segment.end, speaker=label)
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]

def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def merge_speakers(segments: list[TranscriptSegment], turns: list[SpeakerTurn]) -> list[TranscriptSegment]:
    merged = []
    for seg in segments:
        best_turn, best_overlap = None, 0.0
        for turn in turns:
            ov = _overlap(seg.start, seg.end, turn.start, turn.end)
            if ov > best_overlap:
                best_turn, best_overlap = turn, ov
        merged.append(replace(seg, speaker=best_turn.speaker if best_turn else None))
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/extraction/test_diarization.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/extraction/diarization.py backend/tests/extraction/test_diarization.py
git commit -m "feat: add pyannote diarization and speaker-to-segment merge"
```

### Task 7: OCR module (Tesseract)

**Files:**
- Create: `backend/app/extraction/ocr.py`
- Test: `backend/tests/extraction/test_ocr.py`
- Fixture: `backend/tests/fixtures/ocr_sample.png` (an image with visible text, e.g. a screenshot of a slide/diagram with labels)

**Interfaces:**
- Produces: `OCRWord` dataclass (`text: str, bbox: tuple[int, int, int, int], confidence: float`), `ocr_image(image_path: str) -> tuple[str, float, list[OCRWord]]` (returns `(full_text, mean_confidence, words)`) in `backend/app/extraction/ocr.py`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/extraction/test_ocr.py
import os
from app.extraction.ocr import ocr_image

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "ocr_sample.png")

def test_ocr_extracts_text_with_confidence():
    full_text, mean_conf, words = ocr_image(FIXTURE)
    assert len(full_text.strip()) > 0
    assert 0.0 <= mean_conf <= 1.0
    assert len(words) > 0
    assert all(len(w.bbox) == 4 for w in words)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/extraction/test_ocr.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/extraction/ocr.py`**

```python
from dataclasses import dataclass
import pytesseract
from PIL import Image

@dataclass
class OCRWord:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float

def ocr_image(image_path: str) -> tuple[str, float, list[OCRWord]]:
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        words.append(OCRWord(
            text=text,
            bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
            confidence=conf / 100.0,
        ))
    full_text = " ".join(w.text for w in words)
    mean_conf = sum(w.confidence for w in words) / len(words) if words else 0.0
    return full_text, mean_conf, words
```

- [ ] **Step 4: Add fixture and run**

Place any real screenshot with legible text (e.g. a slide with a title and a couple of labels) at `backend/tests/fixtures/ocr_sample.png`.

Run: `cd backend && python -m pytest tests/extraction/test_ocr.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/extraction/ocr.py backend/tests/extraction/test_ocr.py backend/tests/fixtures/ocr_sample.png
git commit -m "feat: add Tesseract OCR wrapper with bbox and confidence"
```

### Task 8: Vision captioning module (GPT-4o-mini)

**Files:**
- Create: `backend/app/extraction/vision.py`
- Test: `backend/tests/extraction/test_vision.py`

**Interfaces:**
- Produces: `ImageCaption` dataclass (`caption: str, description: str, confidence: float`), `caption_image(image_path: str, context: str | None = None) -> ImageCaption` in `backend/app/extraction/vision.py`. Uses OpenAI structured outputs with JSON schema `{caption: string, description: string, confidence: number}`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/extraction/test_vision.py
from unittest.mock import patch, MagicMock
from app.extraction.vision import caption_image, ImageCaption

def test_caption_image_parses_structured_response():
    fake_message = MagicMock()
    fake_message.parsed = {"caption": "Architecture diagram", "description": "A diagram showing a load balancer in front of two app servers connected to a Redis cache and a Postgres database.", "confidence": 0.88}
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    with patch("app.extraction.vision._call_vision_model", return_value=fake_completion):
        result = caption_image("dummy/frame.jpg")
    assert isinstance(result, ImageCaption)
    assert result.caption == "Architecture diagram"
    assert "Redis" in result.description
    assert result.confidence == 0.88
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/extraction/test_vision.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/extraction/vision.py`**

```python
import base64
import os
import json
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class ImageCaption:
    caption: str
    description: str
    confidence: float

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_caption",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
                "description": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["caption", "description", "confidence"],
            "additionalProperties": False,
        },
    },
}

def _call_vision_model(image_path: str, context: str | None):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    prompt = "Describe this image in detail, including any diagrams, architecture, or on-screen text meaning."
    if context:
        prompt += f" Context from surrounding discussion: {context}"
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        response_format=_SCHEMA,
    )

def caption_image(image_path: str, context: str | None = None) -> ImageCaption:
    completion = _call_vision_model(image_path, context)
    parsed = completion.choices[0].message.parsed
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return ImageCaption(**parsed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/extraction/test_vision.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/extraction/vision.py backend/tests/extraction/test_vision.py
git commit -m "feat: add GPT-4o-mini vision captioning wrapper"
```

### Task 9: PDF extraction module (PyMuPDF)

**Files:**
- Create: `backend/app/extraction/pdf.py`
- Test: `backend/tests/extraction/test_pdf.py`
- Fixture: `backend/tests/fixtures/sample.pdf` (a real short PDF with at least one embedded image/diagram)

**Interfaces:**
- Produces: `PDFPage` dataclass (`page_number: int, text: str, image_paths: list[str]`), `extract_pdf(pdf_path: str, out_dir: str) -> list[PDFPage]` in `backend/app/extraction/pdf.py`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/extraction/test_pdf.py
import os
from app.extraction.pdf import extract_pdf

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample.pdf")

def test_extract_pdf_returns_pages_with_text(tmp_path):
    pages = extract_pdf(FIXTURE, str(tmp_path))
    assert len(pages) >= 1
    assert pages[0].page_number == 1
    assert isinstance(pages[0].text, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/extraction/test_pdf.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/extraction/pdf.py`**

```python
import os
from dataclasses import dataclass
import fitz  # PyMuPDF

@dataclass
class PDFPage:
    page_number: int
    text: str
    image_paths: list[str]

def extract_pdf(pdf_path: str, out_dir: str) -> list[PDFPage]:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text()
        image_paths = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_path = os.path.join(out_dir, f"page{page_index + 1}_img{img_index}.png")
            pix.save(image_path)
            image_paths.append(image_path)
        pages.append(PDFPage(page_number=page_index + 1, text=text, image_paths=image_paths))
    return pages
```

- [ ] **Step 4: Add fixture and run**

Place a real short PDF (1-3 pages, at least one with an embedded image or diagram) at `backend/tests/fixtures/sample.pdf`.

Run: `cd backend && python -m pytest tests/extraction/test_pdf.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/extraction/pdf.py backend/tests/extraction/test_pdf.py backend/tests/fixtures/sample.pdf
git commit -m "feat: add PyMuPDF text and embedded-image extraction"
```

### Task 10: NER module (GPT-4o-mini structured output)

**Files:**
- Create: `backend/app/extraction/ner.py`
- Test: `backend/tests/extraction/test_ner.py`

**Interfaces:**
- Produces: `EntityMention` dataclass (`text: str, entity_type: str`), `extract_entities(text: str) -> list[EntityMention]` in `backend/app/extraction/ner.py`. `entity_type` is free-text but constrained by prompt to short lowercase category words (e.g. `technology`, `person`, `organization`, `concept`) — not a DB enum, since real-world entity types are open-ended.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/extraction/test_ner.py
from unittest.mock import patch, MagicMock
from app.extraction.ner import extract_entities, EntityMention

def test_extract_entities_parses_structured_response():
    fake_message = MagicMock()
    fake_message.parsed = {"entities": [
        {"text": "Redis", "entity_type": "technology"},
        {"text": "Rahul", "entity_type": "person"},
    ]}
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    with patch("app.extraction.ner._call_ner_model", return_value=fake_completion):
        entities = extract_entities("Rahul proposed using Redis for caching.")
    assert entities == [EntityMention("Redis", "technology"), EntityMention("Rahul", "person")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/extraction/test_ner.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/extraction/ner.py`**

```python
import os
import json
from dataclasses import dataclass
from openai import OpenAI

@dataclass(eq=True)
class EntityMention:
    text: str
    entity_type: str

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entities",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "entity_type": {"type": "string"},
                        },
                        "required": ["text", "entity_type"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

def _call_ner_model(text: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            "Extract named entities (technologies, people, organizations, concepts, products) "
            "from this text. Use short lowercase entity_type categories.\n\n" + text}],
        response_format=_SCHEMA,
    )

def extract_entities(text: str) -> list[EntityMention]:
    completion = _call_ner_model(text)
    parsed = completion.choices[0].message.parsed
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return [EntityMention(e["text"], e["entity_type"]) for e in parsed["entities"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/extraction/test_ner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/extraction/ner.py backend/tests/extraction/test_ner.py
git commit -m "feat: add GPT-4o-mini structured NER wrapper"
```

### Task 11: Embeddings module

**Files:**
- Create: `backend/app/extraction/embeddings.py`
- Test: `backend/tests/extraction/test_embeddings.py`

**Interfaces:**
- Produces: `embed_texts(texts: list[str]) -> list[list[float]]` in `backend/app/extraction/embeddings.py`, using `text-embedding-3-small` (1536 dims, matching `EMBEDDING_DIM` in `app.storage.models`), batched in groups of up to 100.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/extraction/test_embeddings.py
from unittest.mock import patch, MagicMock
from app.extraction.embeddings import embed_texts

def test_embed_texts_returns_vectors_per_input():
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    with patch("app.extraction.embeddings._call_embeddings", return_value=fake_response):
        vectors = embed_texts(["hello", "world"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/extraction/test_embeddings.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/extraction/embeddings.py`**

```python
import os
from openai import OpenAI

MODEL = "text-embedding-3-small"

def _call_embeddings(batch: list[str]):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.embeddings.create(model=MODEL, input=batch)

def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = _call_embeddings(batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/extraction/test_embeddings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/extraction/embeddings.py backend/tests/extraction/test_embeddings.py
git commit -m "feat: add batched OpenAI embeddings wrapper"
```

---

## Phase 2 — Ingestion orchestration

Ingestion tasks produce **only** `EvidenceUnit` rows (content + modality + provenance + embedding). Entity/Relationship/KnowledgeEvent construction is entirely Phase 3's job, reading EvidenceUnits back out by `source_id` — this keeps ingestion and linking independently testable. Convention: `source_type` records the original file's kind (`video`/`audio`/`image`/`pdf`/`document`); `modality` records the nature of the extracted content (`speech`/`ocr`/`image`/`document`) regardless of source file.

### Task 12: Repository layer — persist EvidenceUnit/Entity/KnowledgeEvent/Relationship

**Files:**
- Create: `backend/app/storage/repository.py`
- Test: `backend/tests/test_repository.py` (integration, requires running Postgres)

**Interfaces:**
- Produces: `model_to_dict(instance) -> dict`, `persist_evidence_unit(session, *, content_text, content_type, modality, source_id, source_type, raw_locator, extraction_method, timestamp_start=None, timestamp_end=None, page_number=None, bbox=None, confidence, confidence_factors=None, event_id=None, metadata_json=None, language="en") -> EvidenceUnit` (embeds `content_text` internally via `app.extraction.embeddings.embed_texts`, sets `id`/`extracted_at`/`searchable_text`), `persist_entity(session, *, canonical_name, entity_type, description=None, aliases=None, metadata_json=None) -> Entity`, `persist_knowledge_event(session, *, event_type, title, summary, timestamp_start=None, timestamp_end=None, confidence, entities=None, metadata_json=None) -> KnowledgeEvent`, `persist_relationship(session, *, source_type, source_id, target_type, target_id, relationship_type, confidence, timestamp_start=None, timestamp_end=None, evidence_ids=None, metadata_json=None) -> Relationship`, `get_evidence_units_by_source(session, source_id) -> list[EvidenceUnit]`, in `backend/app/storage/repository.py`. All persist functions `session.add(instance); await session.flush(); return instance` (no commit — caller controls transaction boundaries).
- Consumes: `embed_texts` (Task 11), ORM models (Task 2).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_repository.py
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.storage.repository import persist_evidence_unit, persist_entity, get_evidence_units_by_source
import uuid

pytestmark = pytest.mark.asyncio

async def test_persist_evidence_unit_embeds_and_saves():
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]):
        async with get_session() as session:
            unit = await persist_evidence_unit(
                session, content_text="Redis reduces DB load", content_type="transcript",
                modality="speech", source_id=source_id, source_type="video",
                raw_locator="video.mp4#t=1,2", extraction_method="whisper-1",
                timestamp_start=1.0, timestamp_end=2.0, confidence=0.9,
            )
            await session.commit()
            assert unit.id is not None
            assert unit.embedding is not None
            fetched = await get_evidence_units_by_source(session, source_id)
            assert len(fetched) == 1
            assert fetched[0].content_text == "Redis reduces DB load"

async def test_persist_entity_defaults():
    with patch("app.storage.repository.embed_texts", return_value=[[0.2] * 1536]):
        async with get_session() as session:
            entity = await persist_entity(session, canonical_name="Redis", entity_type="technology")
            await session.commit()
            assert entity.aliases == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_repository.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/storage/repository.py`**

```python
import uuid
from datetime import datetime
from app.extraction.embeddings import embed_texts
from app.storage.models import EvidenceUnit, Entity, KnowledgeEvent, Relationship

def model_to_dict(instance) -> dict:
    return {c.name: getattr(instance, c.name) for c in instance.__table__.columns}

async def persist_evidence_unit(session, *, content_text, content_type, modality, source_id,
                                 source_type, raw_locator, extraction_method,
                                 timestamp_start=None, timestamp_end=None, page_number=None,
                                 bbox=None, confidence, confidence_factors=None, event_id=None,
                                 metadata_json=None, language="en") -> EvidenceUnit:
    embedding = embed_texts([content_text])[0]
    unit = EvidenceUnit(
        id=uuid.uuid4(), content_text=content_text, content_type=content_type, modality=modality,
        source_id=source_id, source_type=source_type, raw_locator=raw_locator,
        extraction_method=extraction_method, extracted_at=datetime.utcnow(),
        timestamp_start=timestamp_start, timestamp_end=timestamp_end, page_number=page_number,
        bbox=bbox, embedding=embedding, entities=[], keywords=[], confidence=confidence,
        confidence_factors=confidence_factors or {}, event_id=event_id,
        metadata_json=metadata_json or {}, searchable_text=content_text, language=language,
    )
    session.add(unit)
    await session.flush()
    return unit

async def persist_entity(session, *, canonical_name, entity_type, description=None,
                          aliases=None, metadata_json=None) -> Entity:
    embedding = embed_texts([canonical_name])[0]
    entity = Entity(
        id=uuid.uuid4(), canonical_name=canonical_name, entity_type=entity_type,
        description=description, aliases=aliases or [], embedding=embedding,
        metadata_json=metadata_json or {},
    )
    session.add(entity)
    await session.flush()
    return entity

async def persist_knowledge_event(session, *, event_type, title, summary, timestamp_start=None,
                                   timestamp_end=None, confidence, entities=None,
                                   metadata_json=None) -> KnowledgeEvent:
    event = KnowledgeEvent(
        id=uuid.uuid4(), event_type=event_type, title=title, summary=summary,
        timestamp_start=timestamp_start, timestamp_end=timestamp_end, confidence=confidence,
        entities=entities or [], created_at=datetime.utcnow(), metadata_json=metadata_json or {},
    )
    session.add(event)
    await session.flush()
    return event

async def persist_relationship(session, *, source_type, source_id, target_type, target_id,
                                relationship_type, confidence, timestamp_start=None,
                                timestamp_end=None, evidence_ids=None, metadata_json=None) -> Relationship:
    rel = Relationship(
        id=uuid.uuid4(), source_type=source_type, source_id=source_id, target_type=target_type,
        target_id=target_id, relationship_type=relationship_type, confidence=confidence,
        timestamp_start=timestamp_start, timestamp_end=timestamp_end,
        evidence_ids=evidence_ids or [], metadata_json=metadata_json or {},
    )
    session.add(rel)
    await session.flush()
    return rel

async def get_evidence_units_by_source(session, source_id) -> list[EvidenceUnit]:
    from sqlalchemy import select
    result = await session.execute(select(EvidenceUnit).where(EvidenceUnit.source_id == source_id))
    return list(result.scalars().all())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose up -d postgres && cd backend && alembic upgrade head && python -m pytest tests/test_repository.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/storage/repository.py backend/tests/test_repository.py
git commit -m "feat: add repository layer for the four core objects"
```

### Task 13: Video ingestion pipeline

**Files:**
- Create: `backend/app/ingestion/__init__.py`, `backend/app/ingestion/video.py`
- Test: `backend/tests/ingestion/test_video_ingestion.py`

**Interfaces:**
- Produces: `async def ingest_video(session, video_path: str, source_id: uuid.UUID, media_dir: str) -> list[EvidenceUnit]` in `backend/app/ingestion/video.py`.
- Consumes: `extract_audio`, `extract_keyframes` (Task 4); `transcribe` (Task 5); `diarize`, `merge_speakers` (Task 6); `ocr_image` (Task 7); `caption_image` (Task 8); `persist_evidence_unit` (Task 12).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/ingestion/test_video_ingestion.py
import uuid
import pytest
from unittest.mock import patch, MagicMock
from app.storage.db import get_session
from app.ingestion.video import ingest_video
from app.extraction.video import Keyframe
from app.extraction.asr import TranscriptSegment
from app.extraction.diarization import SpeakerTurn
from app.extraction.ocr import OCRWord
from app.extraction.vision import ImageCaption

pytestmark = pytest.mark.asyncio

async def test_ingest_video_creates_speech_ocr_and_caption_units(tmp_path):
    source_id = uuid.uuid4()
    with (
        patch("app.ingestion.video.extract_audio", return_value=str(tmp_path / "a.wav")),
        patch("app.ingestion.video.extract_keyframes", return_value=[Keyframe(5.0, str(tmp_path / "f0.jpg"))]),
        patch("app.ingestion.video.transcribe", return_value=[TranscriptSegment(0.0, 6.0, "We should use Redis", -0.05, 0.01, 0.9)]),
        patch("app.ingestion.video.diarize", return_value=[SpeakerTurn(0.0, 6.0, "SPEAKER_00")]),
        patch("app.ingestion.video.ocr_image", return_value=("REDIS ARCHITECTURE", 0.85, [OCRWord("REDIS", (0,0,10,10), 0.9)])),
        patch("app.ingestion.video.caption_image", return_value=ImageCaption("Architecture diagram", "A diagram of a Redis cache in front of Postgres.", 0.8)),
        patch("app.ingestion.video.embed_texts", return_value=[[0.1] * 1536]),
    ):
        async with get_session() as session:
            units = await ingest_video(session, "dummy.mp4", source_id, str(tmp_path))
            await session.commit()
    modalities = sorted(u.modality for u in units)
    assert modalities == ["image", "ocr", "speech"]
    speech_unit = next(u for u in units if u.modality == "speech")
    assert speech_unit.metadata_json.get("speaker") == "SPEAKER_00"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/ingestion/test_video_ingestion.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/ingestion/video.py`**

```python
import os
import uuid
from app.extraction.video import extract_audio, extract_keyframes
from app.extraction.asr import transcribe
from app.extraction.diarization import diarize, merge_speakers
from app.extraction.ocr import ocr_image
from app.extraction.vision import caption_image
from app.extraction.embeddings import embed_texts  # noqa: F401  (patched in tests; used transitively via repository)
from app.storage.repository import persist_evidence_unit

async def ingest_video(session, video_path: str, source_id: uuid.UUID, media_dir: str) -> list:
    work_dir = os.path.join(media_dir, str(source_id))
    audio_path = extract_audio(video_path, work_dir)
    keyframes = extract_keyframes(video_path, work_dir)

    segments = transcribe(audio_path)
    turns = diarize(audio_path)
    segments = merge_speakers(segments, turns)

    units = []
    for seg in segments:
        unit = await persist_evidence_unit(
            session, content_text=seg.text, content_type="transcript", modality="speech",
            source_id=source_id, source_type="video", raw_locator=f"{video_path}#t={seg.start},{seg.end}",
            extraction_method="whisper-1+pyannote", timestamp_start=seg.start, timestamp_end=seg.end,
            confidence=seg.confidence,
            confidence_factors={"avg_logprob": seg.avg_logprob, "no_speech_prob": seg.no_speech_prob},
            metadata_json={"speaker": seg.speaker},
        )
        units.append(unit)

    for frame in keyframes:
        full_text, mean_conf, _words = ocr_image(frame.image_path)
        if full_text.strip():
            unit = await persist_evidence_unit(
                session, content_text=full_text, content_type="ocr", modality="ocr",
                source_id=source_id, source_type="video", raw_locator=frame.image_path,
                extraction_method="tesseract", timestamp_start=frame.timestamp,
                timestamp_end=frame.timestamp, confidence=mean_conf,
            )
            units.append(unit)

        caption = caption_image(frame.image_path)
        unit = await persist_evidence_unit(
            session, content_text=caption.description, content_type="caption", modality="image",
            source_id=source_id, source_type="video", raw_locator=frame.image_path,
            extraction_method="gpt-4o-mini-vision", timestamp_start=frame.timestamp,
            timestamp_end=frame.timestamp, confidence=caption.confidence,
            metadata_json={"caption": caption.caption},
        )
        units.append(unit)

    return units
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/ingestion/test_video_ingestion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/video.py backend/tests/ingestion/test_video_ingestion.py
git commit -m "feat: add video ingestion pipeline producing speech/ocr/caption evidence units"
```

### Task 14: Audio-only ingestion pipeline

**Files:**
- Create: `backend/app/ingestion/audio.py`
- Test: `backend/tests/ingestion/test_audio_ingestion.py`

**Interfaces:**
- Produces: `async def ingest_audio(session, audio_path: str, source_id: uuid.UUID) -> list[EvidenceUnit]` in `backend/app/ingestion/audio.py`.
- Consumes: `transcribe` (Task 5), `diarize`/`merge_speakers` (Task 6), `persist_evidence_unit` (Task 12).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/ingestion/test_audio_ingestion.py
import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.ingestion.audio import ingest_audio
from app.extraction.asr import TranscriptSegment
from app.extraction.diarization import SpeakerTurn

pytestmark = pytest.mark.asyncio

async def test_ingest_audio_creates_speech_units():
    source_id = uuid.uuid4()
    with (
        patch("app.ingestion.audio.transcribe", return_value=[TranscriptSegment(0.0, 3.0, "Meeting starts at 10am", -0.02, 0.0, 0.95)]),
        patch("app.ingestion.audio.diarize", return_value=[SpeakerTurn(0.0, 3.0, "SPEAKER_00")]),
        patch("app.ingestion.audio.embed_texts", return_value=[[0.1] * 1536]),
    ):
        async with get_session() as session:
            units = await ingest_audio(session, "dummy.wav", source_id)
            await session.commit()
    assert len(units) == 1
    assert units[0].modality == "speech"
    assert units[0].source_type == "audio"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/ingestion/test_audio_ingestion.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/ingestion/audio.py`**

```python
import uuid
from app.extraction.asr import transcribe
from app.extraction.diarization import diarize, merge_speakers
from app.extraction.embeddings import embed_texts  # noqa: F401
from app.storage.repository import persist_evidence_unit

async def ingest_audio(session, audio_path: str, source_id: uuid.UUID) -> list:
    segments = merge_speakers(transcribe(audio_path), diarize(audio_path))
    units = []
    for seg in segments:
        unit = await persist_evidence_unit(
            session, content_text=seg.text, content_type="transcript", modality="speech",
            source_id=source_id, source_type="audio", raw_locator=f"{audio_path}#t={seg.start},{seg.end}",
            extraction_method="whisper-1+pyannote", timestamp_start=seg.start, timestamp_end=seg.end,
            confidence=seg.confidence,
            confidence_factors={"avg_logprob": seg.avg_logprob, "no_speech_prob": seg.no_speech_prob},
            metadata_json={"speaker": seg.speaker},
        )
        units.append(unit)
    return units
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/ingestion/test_audio_ingestion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/audio.py backend/tests/ingestion/test_audio_ingestion.py
git commit -m "feat: add audio-only ingestion pipeline"
```

### Task 15: Image ingestion pipeline

**Files:**
- Create: `backend/app/ingestion/image.py`
- Test: `backend/tests/ingestion/test_image_ingestion.py`

**Interfaces:**
- Produces: `async def ingest_image(session, image_path: str, source_id: uuid.UUID) -> list[EvidenceUnit]` in `backend/app/ingestion/image.py`.
- Consumes: `ocr_image` (Task 7), `caption_image` (Task 8), `persist_evidence_unit` (Task 12).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/ingestion/test_image_ingestion.py
import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.ingestion.image import ingest_image
from app.extraction.ocr import OCRWord
from app.extraction.vision import ImageCaption

pytestmark = pytest.mark.asyncio

async def test_ingest_image_creates_ocr_and_caption_units():
    source_id = uuid.uuid4()
    with (
        patch("app.ingestion.image.ocr_image", return_value=("DIAGRAM TITLE", 0.8, [OCRWord("DIAGRAM", (0,0,5,5), 0.8)])),
        patch("app.ingestion.image.caption_image", return_value=ImageCaption("Diagram", "A network diagram.", 0.7)),
        patch("app.ingestion.image.embed_texts", return_value=[[0.1] * 1536]),
    ):
        async with get_session() as session:
            units = await ingest_image(session, "diagram.png", source_id)
            await session.commit()
    assert sorted(u.modality for u in units) == ["image", "ocr"]
    assert all(u.source_type == "image" for u in units)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/ingestion/test_image_ingestion.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/ingestion/image.py`**

```python
import uuid
from app.extraction.ocr import ocr_image
from app.extraction.vision import caption_image
from app.extraction.embeddings import embed_texts  # noqa: F401
from app.storage.repository import persist_evidence_unit

async def ingest_image(session, image_path: str, source_id: uuid.UUID) -> list:
    units = []
    full_text, mean_conf, _words = ocr_image(image_path)
    if full_text.strip():
        units.append(await persist_evidence_unit(
            session, content_text=full_text, content_type="ocr", modality="ocr",
            source_id=source_id, source_type="image", raw_locator=image_path,
            extraction_method="tesseract", confidence=mean_conf,
        ))
    caption = caption_image(image_path)
    units.append(await persist_evidence_unit(
        session, content_text=caption.description, content_type="caption", modality="image",
        source_id=source_id, source_type="image", raw_locator=image_path,
        extraction_method="gpt-4o-mini-vision", confidence=caption.confidence,
        metadata_json={"caption": caption.caption},
    ))
    return units
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/ingestion/test_image_ingestion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/image.py backend/tests/ingestion/test_image_ingestion.py
git commit -m "feat: add image ingestion pipeline"
```

### Task 16: PDF/document ingestion pipeline

**Files:**
- Create: `backend/app/ingestion/pdf.py`
- Test: `backend/tests/ingestion/test_pdf_ingestion.py`

**Interfaces:**
- Produces: `async def ingest_pdf(session, pdf_path: str, source_id: uuid.UUID, media_dir: str) -> list[EvidenceUnit]` in `backend/app/ingestion/pdf.py`.
- Consumes: `extract_pdf` (Task 9), `ocr_image`/`caption_image` (Tasks 7-8, applied to each embedded image), `persist_evidence_unit` (Task 12).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/ingestion/test_pdf_ingestion.py
import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.ingestion.pdf import ingest_pdf
from app.extraction.pdf import PDFPage
from app.extraction.ocr import OCRWord
from app.extraction.vision import ImageCaption

pytestmark = pytest.mark.asyncio

async def test_ingest_pdf_creates_document_text_and_image_units(tmp_path):
    source_id = uuid.uuid4()
    with (
        patch("app.ingestion.pdf.extract_pdf", return_value=[
            PDFPage(page_number=1, text="Redis reduces load on Postgres.", image_paths=["fig1.png"]),
        ]),
        patch("app.ingestion.pdf.ocr_image", return_value=("FIG 1: CACHE LAYER", 0.75, [OCRWord("FIG", (0,0,5,5), 0.75)])),
        patch("app.ingestion.pdf.caption_image", return_value=ImageCaption("Cache layer figure", "A figure showing a cache layer.", 0.7)),
        patch("app.ingestion.pdf.embed_texts", return_value=[[0.1] * 1536]),
    ):
        async with get_session() as session:
            units = await ingest_pdf(session, "doc.pdf", source_id, str(tmp_path))
            await session.commit()
    content_types = sorted(u.content_type for u in units)
    assert content_types == ["caption", "document_text", "ocr"]
    doc_unit = next(u for u in units if u.content_type == "document_text")
    assert doc_unit.page_number == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/ingestion/test_pdf_ingestion.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/ingestion/pdf.py`**

```python
import uuid
from app.extraction.pdf import extract_pdf
from app.extraction.ocr import ocr_image
from app.extraction.vision import caption_image
from app.extraction.embeddings import embed_texts  # noqa: F401
from app.storage.repository import persist_evidence_unit

async def ingest_pdf(session, pdf_path: str, source_id: uuid.UUID, media_dir: str) -> list:
    pages = extract_pdf(pdf_path, media_dir)
    units = []
    for page in pages:
        if page.text.strip():
            units.append(await persist_evidence_unit(
                session, content_text=page.text, content_type="document_text", modality="document",
                source_id=source_id, source_type="pdf", raw_locator=f"{pdf_path}#page={page.page_number}",
                extraction_method="pymupdf", page_number=page.page_number, confidence=0.98,
            ))
        for image_path in page.image_paths:
            full_text, mean_conf, _words = ocr_image(image_path)
            if full_text.strip():
                units.append(await persist_evidence_unit(
                    session, content_text=full_text, content_type="ocr", modality="ocr",
                    source_id=source_id, source_type="pdf", raw_locator=image_path,
                    extraction_method="tesseract", page_number=page.page_number, confidence=mean_conf,
                ))
            caption = caption_image(image_path)
            units.append(await persist_evidence_unit(
                session, content_text=caption.description, content_type="caption", modality="image",
                source_id=source_id, source_type="pdf", raw_locator=image_path,
                extraction_method="gpt-4o-mini-vision", page_number=page.page_number,
                confidence=caption.confidence, metadata_json={"caption": caption.caption},
            ))
    return units
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/ingestion/test_pdf_ingestion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/pdf.py backend/tests/ingestion/test_pdf_ingestion.py
git commit -m "feat: add PDF ingestion pipeline"
```

### Task 17: Ingestion API + background job orchestrator

**Files:**
- Create: `backend/app/ingestion/orchestrator.py`, `backend/app/api/__init__.py`, `backend/app/api/ingest.py`
- Modify: `backend/app/main.py` (mount router)
- Test: `backend/tests/api/test_ingest_api.py`

**Interfaces:**
- Produces: `JobStatus` dataclass (`job_id: str, status: Literal["pending","running","done","failed"], source_id: str | None, error: str | None`), in-memory `JOBS: dict[str, JobStatus]` and `async def run_ingestion_job(job_id: str, file_path: str, source_type: str, media_dir: str) -> None` in `backend/app/ingestion/orchestrator.py`, dispatching by `source_type` to `ingest_video`/`ingest_audio`/`ingest_image`/`ingest_pdf`. `POST /ingest` (multipart file upload + `source_type` field) returns `{"job_id": str}` and schedules `run_ingestion_job` via FastAPI `BackgroundTasks`; `GET /ingest/{job_id}` returns the `JobStatus`. Both in `backend/app/api/ingest.py`, router included in `backend/app/main.py` as `app.include_router(ingest_router)`.
- Consumes: `ingest_video`/`ingest_audio`/`ingest_image`/`ingest_pdf` (Tasks 13-16), `get_session` (Task 2).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/api/test_ingest_api.py
import io
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

def test_ingest_endpoint_returns_job_id_and_is_pollable():
    client = TestClient(app)
    with patch("app.api.ingest.run_ingestion_job", new=AsyncMock()):
        resp = client.post(
            "/ingest",
            files={"file": ("sample.mp4", io.BytesIO(b"fake bytes"), "video/mp4")},
            data={"source_type": "video"},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        status_resp = client.get(f"/ingest/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] in ("pending", "running", "done")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/api/test_ingest_api.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/ingestion/orchestrator.py`**

```python
import uuid
from dataclasses import dataclass
from typing import Literal, Optional
from app.storage.db import get_session
from app.ingestion.video import ingest_video
from app.ingestion.audio import ingest_audio
from app.ingestion.image import ingest_image
from app.ingestion.pdf import ingest_pdf

@dataclass
class JobStatus:
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    source_id: Optional[str] = None
    error: Optional[str] = None

JOBS: dict[str, JobStatus] = {}

_DISPATCH = {"video": ingest_video, "audio": ingest_audio, "image": ingest_image, "pdf": ingest_pdf}

async def run_ingestion_job(job_id: str, file_path: str, source_type: str, media_dir: str) -> None:
    JOBS[job_id].status = "running"
    source_id = uuid.uuid4()
    JOBS[job_id].source_id = str(source_id)
    try:
        ingest_fn = _DISPATCH[source_type]
        async with get_session() as session:
            if source_type in ("video", "pdf"):
                await ingest_fn(session, file_path, source_id, media_dir)
            else:
                await ingest_fn(session, file_path, source_id)
            await session.commit()
        JOBS[job_id].status = "done"
    except Exception as exc:
        JOBS[job_id].status = "failed"
        JOBS[job_id].error = str(exc)
```

- [ ] **Step 4: Implement `backend/app/api/ingest.py`**

```python
import os
import uuid
from fastapi import APIRouter, UploadFile, Form, BackgroundTasks, HTTPException
from app.ingestion.orchestrator import JOBS, JobStatus, run_ingestion_job

router = APIRouter()
MEDIA_DIR = os.environ.get("MEDIA_STORAGE_DIR", "./data/media")

@router.post("/ingest")
async def ingest(file: UploadFile, source_type: str = Form(...), background_tasks: BackgroundTasks = None):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(job_id=job_id, status="pending")
    os.makedirs(MEDIA_DIR, exist_ok=True)
    file_path = os.path.join(MEDIA_DIR, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    background_tasks.add_task(run_ingestion_job, job_id, file_path, source_type, MEDIA_DIR)
    return {"job_id": job_id}

@router.get("/ingest/{job_id}")
async def ingest_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job not found")
    return JOBS[job_id]
```

- [ ] **Step 5: Mount the router in `backend/app/main.py`**

```python
from fastapi import FastAPI
from app.api.ingest import router as ingest_router

app = FastAPI(title="Multimodal RAG Pipeline")
app.include_router(ingest_router)

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/api/test_ingest_api.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/ingestion/orchestrator.py backend/app/api/ingest.py backend/app/main.py backend/tests/api/test_ingest_api.py
git commit -m "feat: add ingestion API with background job orchestration"
```

---

## Phase 3 — Cross-modal linking

This is the core differentiator (spec §Cross-modal & temporal relationships). Every sub-step (18-23) is a small, independently-testable pure-ish function; Task 25 wires them together against real persisted EvidenceUnits for one source.

### Task 18: Shared cosine similarity util + multi-signal semantic segmentation

**Files:**
- Create: `backend/app/linking/__init__.py`, `backend/app/linking/scoring_utils.py`, `backend/app/linking/segmentation.py`
- Test: `backend/tests/linking/test_segmentation.py`

**Interfaces:**
- Produces: `cosine_similarity(a: list[float], b: list[float]) -> float` in `backend/app/linking/scoring_utils.py` (used by Tasks 18, 19, 21). `SemanticWindow` dataclass (`start: float, end: float, evidence_unit_ids: list[uuid.UUID]`) and `segment_into_windows(speech_units: list[EvidenceUnit], scene_timestamps: list[float] | None = None) -> list[SemanticWindow]` in `backend/app/linking/segmentation.py`. Boundary strength between consecutive speech units: `0.40*similarity_drop + 0.25*speaker_change + 0.20*scene_change + 0.15*pause_norm`, cut when `>= 0.5`. `similarity_drop = 1 - cosine_similarity(prev.embedding, curr.embedding)`; `speaker_change = 1.0` if `prev.metadata_json["speaker"] != curr.metadata_json["speaker"]` else `0.0`; `scene_change = 1.0` if any `scene_timestamps` value falls in `[prev.timestamp_end, curr.timestamp_start]`; `pause_norm = min(1.0, (curr.timestamp_start - prev.timestamp_end) / 2.0)`.
- Consumes: `EvidenceUnit` (Task 2).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/linking/test_segmentation.py
import uuid
from app.storage.models import EvidenceUnit
from app.linking.scoring_utils import cosine_similarity
from app.linking.segmentation import segment_into_windows

def _unit(start, end, speaker, embedding):
    return EvidenceUnit(
        id=uuid.uuid4(), content_text="x", content_type="transcript", modality="speech",
        source_id=uuid.uuid4(), source_type="video", raw_locator="v", extraction_method="whisper-1",
        timestamp_start=start, timestamp_end=end, embedding=embedding, confidence=0.9,
        metadata_json={"speaker": speaker},
    )

def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

def test_segment_merges_same_speaker_similar_content_no_gap():
    units = [
        _unit(0.0, 2.0, "SPEAKER_00", [1.0, 0.0]),
        _unit(2.0, 4.0, "SPEAKER_00", [0.99, 0.01]),
    ]
    windows = segment_into_windows(units)
    assert len(windows) == 1
    assert windows[0].evidence_unit_ids == [units[0].id, units[1].id]

def test_segment_splits_on_speaker_change_and_topic_shift():
    units = [
        _unit(0.0, 2.0, "SPEAKER_00", [1.0, 0.0]),
        _unit(2.0, 4.0, "SPEAKER_01", [0.0, 1.0]),
    ]
    windows = segment_into_windows(units)
    assert len(windows) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/linking/test_segmentation.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/linking/scoring_utils.py`**

```python
import math

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
```

- [ ] **Step 4: Implement `backend/app/linking/segmentation.py`**

```python
from dataclasses import dataclass
import uuid
from app.linking.scoring_utils import cosine_similarity

PAUSE_NORM_SECONDS = 2.0
BOUNDARY_THRESHOLD = 0.5

@dataclass
class SemanticWindow:
    start: float
    end: float
    evidence_unit_ids: list[uuid.UUID]

def _boundary_strength(prev, curr, scene_timestamps: list[float]) -> float:
    similarity_drop = max(0.0, 1.0 - cosine_similarity(prev.embedding, curr.embedding))
    speaker_change = 1.0 if prev.metadata_json.get("speaker") != curr.metadata_json.get("speaker") else 0.0
    gap = max(0.0, curr.timestamp_start - prev.timestamp_end)
    pause_norm = min(1.0, gap / PAUSE_NORM_SECONDS)
    scene_change = 1.0 if any(prev.timestamp_end <= t <= curr.timestamp_start for t in scene_timestamps) else 0.0
    return 0.40 * similarity_drop + 0.25 * speaker_change + 0.20 * scene_change + 0.15 * pause_norm

def segment_into_windows(speech_units: list, scene_timestamps: list[float] | None = None) -> list[SemanticWindow]:
    scene_timestamps = scene_timestamps or []
    ordered = sorted(speech_units, key=lambda u: u.timestamp_start)
    if not ordered:
        return []
    windows: list[SemanticWindow] = []
    current_ids = [ordered[0].id]
    current_start = ordered[0].timestamp_start
    current_end = ordered[0].timestamp_end
    for prev, curr in zip(ordered, ordered[1:]):
        if _boundary_strength(prev, curr, scene_timestamps) >= BOUNDARY_THRESHOLD:
            windows.append(SemanticWindow(current_start, current_end, current_ids))
            current_ids = [curr.id]
            current_start = curr.timestamp_start
        else:
            current_ids.append(curr.id)
        current_end = curr.timestamp_end
    windows.append(SemanticWindow(current_start, current_end, current_ids))
    return windows
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/linking/test_segmentation.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/linking/scoring_utils.py backend/app/linking/segmentation.py backend/tests/linking/test_segmentation.py
git commit -m "feat: add multi-signal semantic segmentation"
```

### Task 19: Candidate scoring funnel (cheap scoring before any LLM call)

**Files:**
- Create: `backend/app/linking/candidate_scoring.py`
- Test: `backend/tests/linking/test_candidate_scoring.py`

**Interfaces:**
- Produces: `score_candidates(window_units: list[EvidenceUnit], candidates: list[EvidenceUnit], top_k: int = 3, timestamp_norm: float = 15.0) -> list[tuple[EvidenceUnit, float]]` in `backend/app/linking/candidate_scoring.py`, sorted descending by score, truncated to `top_k`. Score = `0.35*timestamp_proximity + 0.30*keyword_overlap + 0.35*embedding_similarity`, where `timestamp_proximity = 1 - min(1, distance/timestamp_norm)` (distance = 0 if candidate timestamp falls inside the window span spanned by `window_units`, else distance to nearest edge), `keyword_overlap` = Jaccard similarity of lowercased word sets between the concatenated window text and the candidate's `content_text` (this stands in for true entity overlap, which isn't available until entity resolution runs — Task 21), `embedding_similarity` = `cosine_similarity` between the window's mean-pooled embedding and the candidate's embedding.
- Consumes: `cosine_similarity` (Task 18), `EvidenceUnit` (Task 2).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/linking/test_candidate_scoring.py
import uuid
from app.storage.models import EvidenceUnit
from app.linking.candidate_scoring import score_candidates

def _unit(text, start, end, embedding, modality="speech"):
    return EvidenceUnit(
        id=uuid.uuid4(), content_text=text, content_type="x", modality=modality,
        source_id=uuid.uuid4(), source_type="video", raw_locator="v", extraction_method="x",
        timestamp_start=start, timestamp_end=end, embedding=embedding, confidence=0.9, metadata_json={},
    )

def test_score_candidates_ranks_closer_more_similar_candidate_higher():
    window_units = [_unit("we should use redis architecture for caching", 10.0, 15.0, [1.0, 0.0])]
    good = _unit("REDIS ARCHITECTURE DIAGRAM", 12.0, 12.0, [0.95, 0.05], modality="ocr")
    bad = _unit("unrelated quarterly sales numbers", 90.0, 90.0, [0.0, 1.0], modality="ocr")
    ranked = score_candidates(window_units, [bad, good], top_k=3)
    assert ranked[0][0] is good
    assert ranked[0][1] > ranked[1][1]

def test_score_candidates_respects_top_k():
    window_units = [_unit("topic text", 0.0, 5.0, [1.0, 0.0])]
    candidates = [_unit(f"candidate {i}", float(i), float(i), [1.0, 0.0], modality="ocr") for i in range(5)]
    ranked = score_candidates(window_units, candidates, top_k=2)
    assert len(ranked) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/linking/test_candidate_scoring.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/linking/candidate_scoring.py`**

```python
from app.linking.scoring_utils import cosine_similarity

def _mean_embedding(units) -> list[float]:
    dim = len(units[0].embedding)
    sums = [0.0] * dim
    for u in units:
        for i, v in enumerate(u.embedding):
            sums[i] += v
    return [s / len(units) for s in sums]

def _keyword_overlap(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)

def _timestamp_proximity(window_units, candidate, norm: float) -> float:
    start = min(u.timestamp_start for u in window_units)
    end = max(u.timestamp_end for u in window_units)
    t = candidate.timestamp_start
    distance = 0.0 if start <= t <= end else min(abs(t - start), abs(t - end))
    return 1.0 - min(1.0, distance / norm)

def score_candidates(window_units, candidates, top_k: int = 3, timestamp_norm: float = 15.0):
    window_text = " ".join(u.content_text for u in window_units)
    window_embedding = _mean_embedding(window_units)
    scored = []
    for candidate in candidates:
        timestamp_proximity = _timestamp_proximity(window_units, candidate, timestamp_norm)
        keyword_overlap = _keyword_overlap(window_text, candidate.content_text)
        embedding_similarity = cosine_similarity(window_embedding, candidate.embedding)
        score = 0.35 * timestamp_proximity + 0.30 * keyword_overlap + 0.35 * embedding_similarity
        scored.append((candidate, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/linking/test_candidate_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/linking/candidate_scoring.py backend/tests/linking/test_candidate_scoring.py
git commit -m "feat: add cheap candidate-scoring funnel before LLM relationship calls"
```

### Task 20: Relationship inference (strict-enum LLM call)

**Files:**
- Create: `backend/app/linking/relationship_inference.py`
- Test: `backend/tests/linking/test_relationship_inference.py`

**Interfaces:**
- Produces: `RelationshipInference` dataclass (`related: bool, relationship_type: str, confidence: float, evidence: str`), `async def infer_relationship(window_text: str, candidate_text: str, candidate_modality: str) -> RelationshipInference` in `backend/app/linking/relationship_inference.py`. `relationship_type` constrained via JSON schema enum to exactly the values in Global Constraints.
- Consumes: nothing new (standalone OpenAI structured-output call, same pattern as Task 8/10).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/linking/test_relationship_inference.py
import pytest
from unittest.mock import patch, MagicMock
from app.linking.relationship_inference import infer_relationship, RelationshipInference

pytestmark = pytest.mark.asyncio

async def test_infer_relationship_parses_structured_response():
    fake_message = MagicMock()
    fake_message.parsed = {
        "related": True, "relationship_type": "explains", "confidence": 0.93,
        "evidence": "The speaker describes the Redis architecture visible in the frame",
    }
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    with patch("app.linking.relationship_inference._call_relationship_model", return_value=fake_completion):
        result = await infer_relationship("we use Redis for caching", "REDIS ARCHITECTURE DIAGRAM", "ocr")
    assert isinstance(result, RelationshipInference)
    assert result.related is True
    assert result.relationship_type == "explains"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/linking/test_relationship_inference.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/linking/relationship_inference.py`**

```python
import os
import json
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class RelationshipInference:
    related: bool
    relationship_type: str
    confidence: float
    evidence: str

_RELATIONSHIP_TYPES = ["explains", "supports", "corresponds_to", "mentions", "depicts",
                        "contradicts", "follows", "references", "unrelated"]

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "relationship_inference",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "related": {"type": "boolean"},
                "relationship_type": {"type": "string", "enum": _RELATIONSHIP_TYPES},
                "confidence": {"type": "number"},
                "evidence": {"type": "string"},
            },
            "required": ["related", "relationship_type", "confidence", "evidence"],
            "additionalProperties": False,
        },
    },
}

def _call_relationship_model(window_text: str, candidate_text: str, candidate_modality: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        f"Spoken/topic window text:\n{window_text}\n\n"
        f"Candidate {candidate_modality} evidence:\n{candidate_text}\n\n"
        "Are these related? If so, classify the relationship using exactly one of: "
        + ", ".join(_RELATIONSHIP_TYPES)
    )
    return client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format=_SCHEMA,
    )

async def infer_relationship(window_text: str, candidate_text: str, candidate_modality: str) -> RelationshipInference:
    completion = _call_relationship_model(window_text, candidate_text, candidate_modality)
    parsed = completion.choices[0].message.parsed
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return RelationshipInference(**parsed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/linking/test_relationship_inference.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/linking/relationship_inference.py backend/tests/linking/test_relationship_inference.py
git commit -m "feat: add strict-enum LLM relationship inference"
```

### Task 21: Entity resolution (safe-fallback: alias lookup before embedding search before create)

**Files:**
- Modify: `backend/app/storage/repository.py` (add `list_entities`)
- Create: `backend/app/linking/entity_resolution.py`
- Test: `backend/tests/linking/test_entity_resolution.py`

**Interfaces:**
- Produces: `async def list_entities(session) -> list[Entity]` added to `backend/app/storage/repository.py`. `async def resolve_entity(session, mention_text: str, mention_type: str, threshold: float = 0.85) -> Entity` in `backend/app/linking/entity_resolution.py`: normalize (`.strip().lower()`) → exact match against `canonical_name`/`aliases` (case-insensitive) → embedding search against all entities' `.embedding` via `cosine_similarity`, best match used only if `similarity >= threshold` (appending `mention_text` to its `aliases` if not already present) → otherwise `persist_entity(session, canonical_name=mention_text, entity_type=mention_type, aliases=[])`.
- Consumes: `list_entities`, `persist_entity` (Task 12/this task), `embed_texts` (Task 11), `cosine_similarity` (Task 18).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/linking/test_entity_resolution.py
import uuid
import pytest
from unittest.mock import patch, AsyncMock
from app.storage.models import Entity
from app.linking.entity_resolution import resolve_entity

pytestmark = pytest.mark.asyncio

async def test_resolve_entity_matches_by_alias():
    existing = Entity(id=uuid.uuid4(), canonical_name="PostgreSQL", entity_type="technology",
                       aliases=["postgres", "postgres db"], embedding=[1.0, 0.0])
    with patch("app.linking.entity_resolution.list_entities", new=AsyncMock(return_value=[existing])):
        result = await resolve_entity(None, "Postgres DB", "technology")
    assert result.id == existing.id

async def test_resolve_entity_matches_by_embedding_similarity_above_threshold():
    existing = Entity(id=uuid.uuid4(), canonical_name="Redis", entity_type="technology",
                       aliases=[], embedding=[1.0, 0.0])
    with (
        patch("app.linking.entity_resolution.list_entities", new=AsyncMock(return_value=[existing])),
        patch("app.linking.entity_resolution.embed_texts", return_value=[[0.99, 0.01]]),
    ):
        result = await resolve_entity(None, "redis-cache", "technology", threshold=0.9)
    assert result.id == existing.id
    assert "redis-cache" in existing.aliases

async def test_resolve_entity_creates_new_when_no_match():
    new_entity = Entity(id=uuid.uuid4(), canonical_name="Kafka", entity_type="technology", aliases=[])
    with (
        patch("app.linking.entity_resolution.list_entities", new=AsyncMock(return_value=[])),
        patch("app.linking.entity_resolution.embed_texts", return_value=[[0.0, 1.0]]),
        patch("app.linking.entity_resolution.persist_entity", new=AsyncMock(return_value=new_entity)),
    ):
        result = await resolve_entity(None, "Kafka", "technology")
    assert result.canonical_name == "Kafka"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/linking/test_entity_resolution.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Add `list_entities` to `backend/app/storage/repository.py`**

```python
async def list_entities(session) -> list[Entity]:
    from sqlalchemy import select
    result = await session.execute(select(Entity))
    return list(result.scalars().all())
```

- [ ] **Step 4: Implement `backend/app/linking/entity_resolution.py`**

```python
from app.extraction.embeddings import embed_texts
from app.linking.scoring_utils import cosine_similarity
from app.storage.repository import list_entities, persist_entity

async def resolve_entity(session, mention_text: str, mention_type: str, threshold: float = 0.85):
    normalized = mention_text.strip().lower()
    existing = await list_entities(session)
    for entity in existing:
        if normalized == entity.canonical_name.strip().lower():
            return entity
        if normalized in (a.strip().lower() for a in entity.aliases):
            return entity

    candidates_with_embedding = [e for e in existing if e.embedding is not None]
    if candidates_with_embedding:
        mention_embedding = embed_texts([mention_text])[0]
        best_entity, best_similarity = None, 0.0
        for entity in candidates_with_embedding:
            similarity = cosine_similarity(mention_embedding, entity.embedding)
            if similarity > best_similarity:
                best_entity, best_similarity = entity, similarity
        if best_entity is not None and best_similarity >= threshold:
            if mention_text not in best_entity.aliases:
                best_entity.aliases = best_entity.aliases + [mention_text]
            return best_entity

    return await persist_entity(session, canonical_name=mention_text, entity_type=mention_type, aliases=[])
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/linking/test_entity_resolution.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/storage/repository.py backend/app/linking/entity_resolution.py backend/tests/linking/test_entity_resolution.py
git commit -m "feat: add safe-fallback entity resolution (alias -> embedding -> create)"
```

### Task 22: KnowledgeEvent assembly (deterministic EventScore)

**Files:**
- Create: `backend/app/linking/event_assembly.py`
- Test: `backend/tests/linking/test_event_assembly.py`

**Interfaces:**
- Produces: `EVENT_SCORE_THRESHOLD = 0.5`, `EVENT_TYPES` (the 12-value set from Global Constraints), `compute_event_score(temporal_overlap: float, entity_overlap: float, semantic_similarity: float, relationship_confidence: float) -> float` (= `0.35*temporal_overlap + 0.30*entity_overlap + 0.20*semantic_similarity + 0.15*relationship_confidence`), and `async def assemble_knowledge_event(session, *, event_type: str, title: str, summary: str, timestamp_start, timestamp_end, temporal_overlap: float, entity_overlap: float, semantic_similarity: float, relationship_confidence: float, member_units: list, entity_ids: list | None = None) -> KnowledgeEvent | None` in `backend/app/linking/event_assembly.py`. Returns `None` (no event created) when the computed score is below `EVENT_SCORE_THRESHOLD`; otherwise persists the event via `persist_knowledge_event` and sets `.event_id` on every unit in `member_units`.
- Consumes: `persist_knowledge_event` (Task 12).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/linking/test_event_assembly.py
import uuid
import pytest
from unittest.mock import patch, AsyncMock
from app.storage.models import KnowledgeEvent
from app.linking.event_assembly import compute_event_score, assemble_knowledge_event, EVENT_SCORE_THRESHOLD

pytestmark = pytest.mark.asyncio

def test_compute_event_score_matches_spec_formula():
    score = compute_event_score(temporal_overlap=0.8, entity_overlap=0.6,
                                 semantic_similarity=0.9, relationship_confidence=0.7)
    assert score == pytest.approx(0.35 * 0.8 + 0.30 * 0.6 + 0.20 * 0.9 + 0.15 * 0.7)

async def test_assemble_knowledge_event_creates_and_links_units():
    fake_event = KnowledgeEvent(id=uuid.uuid4(), event_type="diagram_explanation", title="t",
                                 summary="s", timestamp_start=0.0, timestamp_end=5.0, confidence=0.9,
                                 entities=[], metadata_json={})
    class FakeUnit:
        event_id = None
    unit = FakeUnit()
    with patch("app.linking.event_assembly.persist_knowledge_event", new=AsyncMock(return_value=fake_event)):
        result = await assemble_knowledge_event(
            None, event_type="diagram_explanation", title="t", summary="s",
            timestamp_start=0.0, timestamp_end=5.0, temporal_overlap=1.0, entity_overlap=1.0,
            semantic_similarity=1.0, relationship_confidence=1.0, member_units=[unit],
        )
    assert result is fake_event
    assert unit.event_id == fake_event.id

async def test_assemble_knowledge_event_returns_none_below_threshold():
    result = await assemble_knowledge_event(
        None, event_type="discussion", title="t", summary="s", timestamp_start=0.0, timestamp_end=1.0,
        temporal_overlap=0.1, entity_overlap=0.1, semantic_similarity=0.1, relationship_confidence=0.1,
        member_units=[],
    )
    assert result is None
    assert 0.1 * 0.35 + 0.1 * 0.30 + 0.1 * 0.20 + 0.1 * 0.15 < EVENT_SCORE_THRESHOLD
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/linking/test_event_assembly.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/linking/event_assembly.py`**

```python
from app.storage.repository import persist_knowledge_event

EVENT_SCORE_THRESHOLD = 0.5
EVENT_TYPES = {
    "discussion", "decision", "proposal", "explanation", "diagram_explanation", "definition",
    "question", "answer", "comparison", "action_item", "observation", "reference",
}

def compute_event_score(temporal_overlap: float, entity_overlap: float,
                         semantic_similarity: float, relationship_confidence: float) -> float:
    return (0.35 * temporal_overlap + 0.30 * entity_overlap
            + 0.20 * semantic_similarity + 0.15 * relationship_confidence)

async def assemble_knowledge_event(session, *, event_type: str, title: str, summary: str,
                                    timestamp_start, timestamp_end, temporal_overlap: float,
                                    entity_overlap: float, semantic_similarity: float,
                                    relationship_confidence: float, member_units: list,
                                    entity_ids: list | None = None):
    assert event_type in EVENT_TYPES, f"invalid event_type: {event_type}"
    score = compute_event_score(temporal_overlap, entity_overlap, semantic_similarity, relationship_confidence)
    if score < EVENT_SCORE_THRESHOLD:
        return None
    event = await persist_knowledge_event(
        session, event_type=event_type, title=title, summary=summary,
        timestamp_start=timestamp_start, timestamp_end=timestamp_end,
        confidence=min(1.0, score), entities=entity_ids or [],
    )
    for unit in member_units:
        unit.event_id = event.id
    return event
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/linking/test_event_assembly.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/linking/event_assembly.py backend/tests/linking/test_event_assembly.py
git commit -m "feat: add deterministic EventScore-based KnowledgeEvent assembly"
```

### Task 23: Contradiction detection between chronologically-ordered events on the same entity

**Files:**
- Create: `backend/app/linking/contradiction.py`
- Test: `backend/tests/linking/test_contradiction.py`

**Interfaces:**
- Produces: `ContradictionResult` dataclass (`contradicts: bool, confidence: float, explanation: str`), `async def check_contradiction(earlier_summary: str, later_summary: str) -> ContradictionResult`, and `async def detect_and_record_contradictions(session, events_for_entity: list[KnowledgeEvent]) -> list[Relationship]` in `backend/app/linking/contradiction.py`. The latter sorts `events_for_entity` by `timestamp_start`, checks each chronologically-consecutive pair (O(n), not all pairs), and for every `contradicts=True` result with `confidence >= 0.6` persists a `Relationship(relationship_type="contradicts", source_id=earlier.id, target_id=later.id, metadata_json={"superseded_by": str(later.id)})` via `persist_relationship`.
- Consumes: `persist_relationship` (Task 12).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/linking/test_contradiction.py
import uuid
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.storage.models import KnowledgeEvent, Relationship
from app.linking.contradiction import check_contradiction, detect_and_record_contradictions, ContradictionResult

pytestmark = pytest.mark.asyncio

async def test_check_contradiction_parses_structured_response():
    fake_message = MagicMock()
    fake_message.parsed = {"contradicts": True, "confidence": 0.9, "explanation": "Later decision reverses the earlier proposal"}
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    with patch("app.linking.contradiction._call_contradiction_model", return_value=fake_completion):
        result = await check_contradiction("We'll use Redis.", "Actually, we decided not to use Redis.")
    assert isinstance(result, ContradictionResult)
    assert result.contradicts is True

async def test_detect_and_record_contradictions_persists_relationship_for_confirmed_pair():
    earlier = KnowledgeEvent(id=uuid.uuid4(), event_type="proposal", title="t1", summary="We'll use Redis.",
                              timestamp_start=0.0, timestamp_end=5.0, confidence=0.9, entities=[], metadata_json={})
    later = KnowledgeEvent(id=uuid.uuid4(), event_type="decision", title="t2", summary="We decided not to use Redis.",
                            timestamp_start=100.0, timestamp_end=105.0, confidence=0.9, entities=[], metadata_json={})
    fake_result = ContradictionResult(contradicts=True, confidence=0.85, explanation="reversal")
    fake_relationship = Relationship(id=uuid.uuid4(), source_type="knowledge_event", source_id=earlier.id,
                                      target_type="knowledge_event", target_id=later.id,
                                      relationship_type="contradicts", confidence=0.85, evidence_ids=[], metadata_json={})
    with (
        patch("app.linking.contradiction.check_contradiction", new=AsyncMock(return_value=fake_result)),
        patch("app.linking.contradiction.persist_relationship", new=AsyncMock(return_value=fake_relationship)) as mock_persist,
    ):
        created = await detect_and_record_contradictions(None, [later, earlier])
    assert len(created) == 1
    mock_persist.assert_awaited_once()
    _, kwargs = mock_persist.call_args
    assert kwargs["source_id"] == earlier.id and kwargs["target_id"] == later.id
    assert kwargs["metadata_json"]["superseded_by"] == str(later.id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/linking/test_contradiction.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/linking/contradiction.py`**

```python
import os
import json
from dataclasses import dataclass
from openai import OpenAI
from app.storage.repository import persist_relationship

@dataclass
class ContradictionResult:
    contradicts: bool
    confidence: float
    explanation: str

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "contradiction_check", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "contradicts": {"type": "boolean"},
                "confidence": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": ["contradicts", "confidence", "explanation"],
            "additionalProperties": False,
        },
    },
}

def _call_contradiction_model(earlier_summary: str, later_summary: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        f"Earlier statement: {earlier_summary}\nLater statement: {later_summary}\n\n"
        "Does the later statement contradict or reverse the earlier one about the same subject?"
    )
    return client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format=_SCHEMA,
    )

async def check_contradiction(earlier_summary: str, later_summary: str) -> ContradictionResult:
    completion = _call_contradiction_model(earlier_summary, later_summary)
    parsed = completion.choices[0].message.parsed
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return ContradictionResult(**parsed)

async def detect_and_record_contradictions(session, events_for_entity: list) -> list:
    ordered = sorted(events_for_entity, key=lambda e: e.timestamp_start)
    created = []
    for earlier, later in zip(ordered, ordered[1:]):
        result = await check_contradiction(earlier.summary, later.summary)
        if result.contradicts and result.confidence >= 0.6:
            rel = await persist_relationship(
                session, source_type="knowledge_event", source_id=earlier.id,
                target_type="knowledge_event", target_id=later.id, relationship_type="contradicts",
                confidence=result.confidence, timestamp_start=earlier.timestamp_start,
                timestamp_end=later.timestamp_end, evidence_ids=[],
                metadata_json={"superseded_by": str(later.id), "explanation": result.explanation},
            )
            created.append(rel)
    return created
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/linking/test_contradiction.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/linking/contradiction.py backend/tests/linking/test_contradiction.py
git commit -m "feat: add chronological contradiction/supersession detection"
```

### Task 24: Graph sync — materialize Postgres rows into Neo4j

**Files:**
- Create: `backend/app/linking/graph_sync.py`
- Test: `backend/tests/linking/test_graph_sync.py` (integration, requires running Neo4j)

**Interfaces:**
- Produces: `async def sync_to_graph(driver, *, entities: list, evidence_units: list, events: list, relationships: list) -> None` in `backend/app/linking/graph_sync.py`. Upserts every node/edge via Task 3's helpers (using `model_to_dict`), then adds `(:EvidenceUnit)-[:BELONGS_TO]->(:KnowledgeEvent)` for every unit with a non-null `event_id`, and `(:EvidenceUnit)-[:MENTIONS]->(:Entity)` for every id in a unit's `entities` list.
- Consumes: `upsert_entity_node`, `upsert_evidence_node`, `upsert_event_node`, `upsert_relationship_edge` (Task 3), `model_to_dict` (Task 12).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/linking/test_graph_sync.py
import uuid
import pytest
from app.storage.graph import get_driver, setup_constraints
from app.storage.models import Entity, EvidenceUnit, KnowledgeEvent
from app.linking.graph_sync import sync_to_graph

pytestmark = pytest.mark.asyncio

async def test_sync_to_graph_creates_belongs_to_and_mentions_edges():
    driver = get_driver()
    await setup_constraints(driver)
    entity = Entity(id=uuid.uuid4(), canonical_name="Redis", entity_type="technology",
                     description=None, aliases=[], metadata_json={})
    event = KnowledgeEvent(id=uuid.uuid4(), event_type="explanation", title="t", summary="s",
                            timestamp_start=0.0, timestamp_end=5.0, confidence=0.9, entities=[], metadata_json={})
    unit = EvidenceUnit(id=uuid.uuid4(), content_text="x", content_type="transcript", modality="speech",
                         source_id=uuid.uuid4(), source_type="video", raw_locator="v",
                         extraction_method="whisper-1", timestamp_start=0.0, timestamp_end=1.0,
                         confidence=0.9, entities=[entity.id], event_id=event.id, metadata_json={})

    await sync_to_graph(driver, entities=[entity], evidence_units=[unit], events=[event], relationships=[])

    async with driver.session() as session:
        result = await session.run(
            "MATCH (v:EvidenceUnit {id:$vid})-[:BELONGS_TO]->(e:KnowledgeEvent {id:$eid}) RETURN count(*) AS c",
            vid=str(unit.id), eid=str(event.id),
        )
        assert (await result.single())["c"] == 1
        result = await session.run(
            "MATCH (v:EvidenceUnit {id:$vid})-[:MENTIONS]->(en:Entity {id:$enid}) RETURN count(*) AS c",
            vid=str(unit.id), enid=str(entity.id),
        )
        assert (await result.single())["c"] == 1
    await driver.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/linking/test_graph_sync.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/linking/graph_sync.py`**

```python
from app.storage.graph import upsert_entity_node, upsert_evidence_node, upsert_event_node, upsert_relationship_edge
from app.storage.repository import model_to_dict

async def sync_to_graph(driver, *, entities: list, evidence_units: list, events: list, relationships: list) -> None:
    async with driver.session() as session:
        for entity in entities:
            await session.execute_write(upsert_entity_node, model_to_dict(entity))
        for event in events:
            await session.execute_write(upsert_event_node, model_to_dict(event))
        for unit in evidence_units:
            await session.execute_write(upsert_evidence_node, model_to_dict(unit))
        for rel in relationships:
            await session.execute_write(upsert_relationship_edge, model_to_dict(rel))
        for unit in evidence_units:
            if unit.event_id:
                await session.run(
                    "MATCH (v:EvidenceUnit {id:$vid}), (e:KnowledgeEvent {id:$eid}) MERGE (v)-[:BELONGS_TO]->(e)",
                    vid=str(unit.id), eid=str(unit.event_id),
                )
            for entity_id in unit.entities:
                await session.run(
                    "MATCH (v:EvidenceUnit {id:$vid}), (en:Entity {id:$enid}) MERGE (v)-[:MENTIONS]->(en)",
                    vid=str(unit.id), enid=str(entity_id),
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose up -d neo4j && cd backend && python -m pytest tests/linking/test_graph_sync.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/linking/graph_sync.py backend/tests/linking/test_graph_sync.py
git commit -m "feat: sync EvidenceUnit/Entity/KnowledgeEvent/Relationship rows into Neo4j"
```

### Task 25: Linking orchestrator — wire steps A-F together for one ingested source

**Files:**
- Create: `backend/app/linking/summarization.py`, `backend/app/linking/orchestrator.py`
- Modify: `backend/app/api/ingest.py` (call the linking orchestrator after ingestion succeeds)
- Test: `backend/tests/linking/test_orchestrator.py`

**Interfaces:**
- Produces: `EventSummary` dataclass (`event_type: str, title: str, summary: str`), `async def summarize_event(evidence_texts: list[str]) -> EventSummary` in `backend/app/linking/summarization.py` (OpenAI structured output, `event_type` enum-constrained to `EVENT_TYPES` from Task 22). `async def link_source(session, driver, source_id) -> dict` in `backend/app/linking/orchestrator.py`, returning `{"windows": int, "events_created": int, "contradictions_found": int}`. Runs, per source: (1) load EvidenceUnits, split speech vs. candidate (ocr/image/document); (2) `segment_into_windows` using candidate timestamps as `scene_timestamps`; (3) per window: `score_candidates` -> `infer_relationship` on each -> keep `related=True`; (4) NER + `resolve_entity` for every unit's `content_text`, appending resolved ids to `unit.entities`; (5) per-window `entity_overlap` = Jaccard of speech-unit entity ids vs. accepted-candidate entity ids; `semantic_similarity`/`relationship_confidence` = mean of accepted relationships' confidence (default `1.0` if none, matching the single-modality-is-still-valid rule); `summarize_event` over the window's + accepted candidates' `content_text` -> `assemble_knowledge_event`; (6) any EvidenceUnit still without an `event_id` after all windows is wrapped into its own single-modality event (`temporal_overlap=entity_overlap=semantic_similarity=relationship_confidence=1.0`); (7) group resulting KnowledgeEvents by shared entity id, run `detect_and_record_contradictions` per group with >1 event; (8) `sync_to_graph` with everything touched.
- Consumes: everything from Tasks 12, 13-17 (indirectly, via persisted data), 18-24.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/linking/test_orchestrator.py
import uuid
import pytest
from unittest.mock import patch, AsyncMock
from app.linking.orchestrator import link_source
from app.storage.models import EvidenceUnit

pytestmark = pytest.mark.asyncio

def _unit(text, modality, start, end, embedding, source_type="video"):
    return EvidenceUnit(
        id=uuid.uuid4(), content_text=text, content_type="x", modality=modality,
        source_id=uuid.uuid4(), source_type=source_type, raw_locator="v", extraction_method="x",
        timestamp_start=start, timestamp_end=end, embedding=embedding, confidence=0.9,
        entities=[], metadata_json={},
    )

async def test_link_source_creates_multimodal_event_from_related_speech_and_frame():
    speech = _unit("we should use redis for caching", "speech", 0.0, 5.0, [1.0, 0.0])
    frame = _unit("REDIS ARCHITECTURE", "ocr", 2.0, 2.0, [0.98, 0.02])

    from app.linking.relationship_inference import RelationshipInference
    from app.linking.summarization import EventSummary
    from app.storage.models import KnowledgeEvent

    fake_event = KnowledgeEvent(id=uuid.uuid4(), event_type="diagram_explanation", title="Redis caching",
                                 summary="Redis proposed for caching", timestamp_start=0.0, timestamp_end=5.0,
                                 confidence=0.9, entities=[], metadata_json={})

    with (
        patch("app.linking.orchestrator.get_evidence_units_by_source", new=AsyncMock(return_value=[speech, frame])),
        patch("app.linking.orchestrator.infer_relationship", new=AsyncMock(
            return_value=RelationshipInference(True, "explains", 0.9, "matches"))),
        patch("app.linking.orchestrator.extract_entities", return_value=[]),
        patch("app.linking.orchestrator.summarize_event", new=AsyncMock(
            return_value=EventSummary("diagram_explanation", "Redis caching", "Redis proposed for caching"))),
        patch("app.linking.orchestrator.assemble_knowledge_event", new=AsyncMock(return_value=fake_event)),
        patch("app.linking.orchestrator.detect_and_record_contradictions", new=AsyncMock(return_value=[])),
        patch("app.linking.orchestrator.sync_to_graph", new=AsyncMock()),
    ):
        result = await link_source(None, None, uuid.uuid4())

    assert result["events_created"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/linking/test_orchestrator.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/linking/summarization.py`**

```python
import os
import json
from dataclasses import dataclass
from openai import OpenAI
from app.linking.event_assembly import EVENT_TYPES

@dataclass
class EventSummary:
    event_type: str
    title: str
    summary: str

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "event_summary", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "event_type": {"type": "string", "enum": sorted(EVENT_TYPES)},
                "title": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["event_type", "title", "summary"],
            "additionalProperties": False,
        },
    },
}

def _call_summary_model(evidence_texts: list[str]):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    joined = "\n---\n".join(evidence_texts)
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            "Summarize the following connected pieces of evidence as one knowledge event. "
            "Pick the single best event_type.\n\n" + joined}],
        response_format=_SCHEMA,
    )

async def summarize_event(evidence_texts: list[str]) -> EventSummary:
    completion = _call_summary_model(evidence_texts)
    parsed = completion.choices[0].message.parsed
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return EventSummary(**parsed)
```

- [ ] **Step 4: Implement `backend/app/linking/orchestrator.py`**

```python
from collections import defaultdict
from app.storage.repository import get_evidence_units_by_source
from app.linking.segmentation import segment_into_windows
from app.linking.candidate_scoring import score_candidates
from app.linking.relationship_inference import infer_relationship
from app.linking.entity_resolution import resolve_entity
from app.linking.event_assembly import assemble_knowledge_event
from app.linking.summarization import summarize_event
from app.linking.contradiction import detect_and_record_contradictions
from app.linking.graph_sync import sync_to_graph
from app.extraction.ner import extract_entities

async def _resolve_unit_entities(session, unit) -> None:
    mentions = extract_entities(unit.content_text)
    for mention in mentions:
        entity = await resolve_entity(session, mention.text, mention.entity_type)
        if entity.id not in unit.entities:
            unit.entities = unit.entities + [entity.id]

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

async def link_source(session, driver, source_id) -> dict:
    units = await get_evidence_units_by_source(session, source_id)
    speech_units = [u for u in units if u.modality == "speech"]
    candidate_units = [u for u in units if u.modality in ("ocr", "image", "document")]

    for unit in units:
        await _resolve_unit_entities(session, unit)

    scene_timestamps = sorted({u.timestamp_start for u in candidate_units if u.timestamp_start is not None})
    windows = segment_into_windows(speech_units, scene_timestamps)

    events = []
    linked_candidate_ids = set()
    for window in windows:
        window_units = [u for u in speech_units if u.id in window.evidence_unit_ids]
        window_candidates = [
            u for u in candidate_units
            if u.timestamp_start is not None and window.start - 15 <= u.timestamp_start <= window.end + 15
        ]
        ranked = score_candidates(window_units, window_candidates, top_k=3) if window_candidates else []

        accepted = []
        for candidate, _score in ranked:
            window_text = " ".join(u.content_text for u in window_units)
            inference = await infer_relationship(window_text, candidate.content_text, candidate.modality)
            if inference.related:
                accepted.append((candidate, inference))
                linked_candidate_ids.add(candidate.id)

        speech_entities = {eid for u in window_units for eid in u.entities}
        candidate_entities = {eid for c, _ in accepted for eid in c.entities}
        entity_overlap = _jaccard(speech_entities, candidate_entities) if accepted else 1.0
        semantic_similarity = (sum(r.confidence for _, r in accepted) / len(accepted)) if accepted else 1.0
        relationship_confidence = semantic_similarity

        evidence_texts = [u.content_text for u in window_units] + [c.content_text for c, _ in accepted]
        summary = await summarize_event(evidence_texts)

        event = await assemble_knowledge_event(
            session, event_type=summary.event_type, title=summary.title, summary=summary.summary,
            timestamp_start=window.start, timestamp_end=window.end, temporal_overlap=1.0,
            entity_overlap=entity_overlap, semantic_similarity=semantic_similarity,
            relationship_confidence=relationship_confidence,
            member_units=window_units + [c for c, _ in accepted],
            entity_ids=list(speech_entities | candidate_entities),
        )
        if event:
            events.append(event)

    leftover = [u for u in units if u.event_id is None]
    for unit in leftover:
        summary = await summarize_event([unit.content_text])
        event = await assemble_knowledge_event(
            session, event_type=summary.event_type, title=summary.title, summary=summary.summary,
            timestamp_start=unit.timestamp_start, timestamp_end=unit.timestamp_end,
            temporal_overlap=1.0, entity_overlap=1.0, semantic_similarity=1.0, relationship_confidence=1.0,
            member_units=[unit], entity_ids=list(unit.entities),
        )
        if event:
            events.append(event)

    entity_to_events = defaultdict(list)
    for event in events:
        for entity_id in event.entities:
            entity_to_events[entity_id].append(event)

    contradictions = []
    for entity_id, entity_events in entity_to_events.items():
        if len(entity_events) > 1:
            contradictions.extend(await detect_and_record_contradictions(session, entity_events))

    all_entities = []  # populated by caller if it needs the full entity set; graph sync only needs touched entities
    await sync_to_graph(driver, entities=all_entities, evidence_units=units, events=events, relationships=contradictions)

    return {"windows": len(windows), "events_created": len(events), "contradictions_found": len(contradictions)}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/linking/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 6: Wire into the ingestion job in `backend/app/api/ingest.py` / `backend/app/ingestion/orchestrator.py`**

In `backend/app/ingestion/orchestrator.py`, after `await session.commit()` inside `run_ingestion_job`, add a call to link the freshly-ingested source:

```python
from app.linking.orchestrator import link_source
from app.storage.graph import get_driver
...
        JOBS[job_id].status = "done"
        driver = get_driver()
        async with get_session() as link_session:
            await link_source(link_session, driver, source_id)
            await link_session.commit()
        await driver.close()
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/linking/summarization.py backend/app/linking/orchestrator.py backend/app/ingestion/orchestrator.py backend/tests/linking/test_orchestrator.py
git commit -m "feat: wire segmentation/scoring/relationship/entity/event/contradiction into linking orchestrator"
```

---

## Phase 4 — Retrieval

### Task 26: Vector search over EvidenceUnits and KnowledgeEvents

**Files:**
- Modify: `backend/app/storage/models.py` (add `embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)` to `KnowledgeEvent`)
- Modify: `backend/app/storage/repository.py` (`persist_knowledge_event` now also computes and stores `embedding = embed_texts([summary])[0]`)
- Modify: `backend/migrations/versions/` (new Alembic revision adding the column)
- Create: `backend/app/retrieval/__init__.py`, `backend/app/retrieval/vector_search.py`
- Test: `backend/tests/retrieval/test_vector_search.py` (integration, requires running Postgres)

**Interfaces:**
- Produces: `async def vector_search_evidence(session, query_embedding: list[float], top_k: int = 10) -> list[tuple[EvidenceUnit, float]]` and `async def vector_search_events(session, query_embedding: list[float], top_k: int = 10) -> list[tuple[KnowledgeEvent, float]]` in `backend/app/retrieval/vector_search.py`, both returning `(row, similarity)` pairs sorted by similarity descending, using pgvector's `cosine_distance` (`similarity = 1 - distance`).
- Consumes: `EvidenceUnit`, `KnowledgeEvent` (Task 2/this task).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/retrieval/test_vector_search.py
import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.storage.repository import persist_evidence_unit, persist_knowledge_event
from app.retrieval.vector_search import vector_search_evidence, vector_search_events

pytestmark = pytest.mark.asyncio

async def test_vector_search_evidence_ranks_closest_first():
    source_id = uuid.uuid4()
    with patch("app.storage.repository.embed_texts") as mock_embed:
        async with get_session() as session:
            mock_embed.return_value = [[1.0] + [0.0] * 1535]
            close = await persist_evidence_unit(
                session, content_text="close", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            mock_embed.return_value = [[0.0, 1.0] + [0.0] * 1534]
            far = await persist_evidence_unit(
                session, content_text="far", content_type="transcript", modality="speech",
                source_id=source_id, source_type="video", raw_locator="v", extraction_method="x",
                confidence=0.9,
            )
            await session.commit()
            results = await vector_search_evidence(session, [1.0] + [0.0] * 1535, top_k=5)
    ids_in_order = [r[0].id for r in results]
    assert ids_in_order.index(close.id) < ids_in_order.index(far.id)

async def test_vector_search_events_returns_similarity_scores():
    with patch("app.storage.repository.embed_texts", return_value=[[1.0] + [0.0] * 1535]):
        async with get_session() as session:
            event = await persist_knowledge_event(
                session, event_type="discussion", title="t", summary="Redis caching",
                timestamp_start=0.0, timestamp_end=1.0, confidence=0.9,
            )
            await session.commit()
            results = await vector_search_events(session, [1.0] + [0.0] * 1535, top_k=5)
    assert any(r[0].id == event.id for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/retrieval/test_vector_search.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Add `embedding` column to `KnowledgeEvent` in `backend/app/storage/models.py`**

```python
class KnowledgeEvent(Base):
    __tablename__ = "knowledge_events"
    id: Mapped[uuid.UUID] = _uuid_col()
    event_type: Mapped[EventType] = mapped_column(SAEnum(EventType, name="event_type_enum"))
    title: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(String)
    timestamp_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    entities: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_units: Mapped[list["EvidenceUnit"]] = relationship(back_populates="event")
```

- [ ] **Step 4: Update `persist_knowledge_event` in `backend/app/storage/repository.py`**

```python
async def persist_knowledge_event(session, *, event_type, title, summary, timestamp_start=None,
                                   timestamp_end=None, confidence, entities=None,
                                   metadata_json=None) -> KnowledgeEvent:
    embedding = embed_texts([summary])[0]
    event = KnowledgeEvent(
        id=uuid.uuid4(), event_type=event_type, title=title, summary=summary,
        timestamp_start=timestamp_start, timestamp_end=timestamp_end, embedding=embedding,
        confidence=confidence, entities=entities or [], created_at=datetime.utcnow(),
        metadata_json=metadata_json or {},
    )
    session.add(event)
    await session.flush()
    return event
```

- [ ] **Step 5: Generate and apply the Alembic migration**

Run: `docker compose up -d postgres && cd backend && alembic revision --autogenerate -m "add knowledge_events.embedding" && alembic upgrade head`

- [ ] **Step 6: Implement `backend/app/retrieval/vector_search.py`**

```python
from sqlalchemy import select
from app.storage.models import EvidenceUnit, KnowledgeEvent

async def vector_search_evidence(session, query_embedding: list[float], top_k: int = 10):
    distance = EvidenceUnit.embedding.cosine_distance(query_embedding)
    result = await session.execute(
        select(EvidenceUnit, distance.label("distance"))
        .where(EvidenceUnit.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    return [(row.EvidenceUnit, 1.0 - row.distance) for row in result]

async def vector_search_events(session, query_embedding: list[float], top_k: int = 10):
    distance = KnowledgeEvent.embedding.cosine_distance(query_embedding)
    result = await session.execute(
        select(KnowledgeEvent, distance.label("distance"))
        .where(KnowledgeEvent.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    return [(row.KnowledgeEvent, 1.0 - row.distance) for row in result]
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/retrieval/test_vector_search.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/app/storage/models.py backend/app/storage/repository.py backend/migrations backend/app/retrieval/vector_search.py backend/tests/retrieval/test_vector_search.py
git commit -m "feat: add pgvector search over evidence units and knowledge events"
```

### Task 27: Entity-path retrieval + graph expansion

**Files:**
- Create: `backend/app/retrieval/graph_expansion.py`
- Test: `backend/tests/retrieval/test_graph_expansion.py` (integration, requires running Neo4j)

**Interfaces:**
- Produces: `async def entities_for_query(session, query_text: str) -> list` (reuses `extract_entities` + `resolve_entity`, read path — may still create new entities for unmatched query terms, which is acceptable since they simply won't match anything in the graph), `async def evidence_linked_to_entities(driver, entity_ids: list) -> list[str]` (returns `EvidenceUnit.id` strings connected via `MENTIONS`), `async def graph_expand(driver, seed_ids: list[str], hops: int = 2) -> list[str]` (returns node id strings reachable from `seed_ids` within `hops` hops over `RELATES`/`BELONGS_TO`/`MENTIONS` edges) — all in `backend/app/retrieval/graph_expansion.py`.
- Consumes: `extract_entities` (Task 10), `resolve_entity` (Task 21).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/retrieval/test_graph_expansion.py
import uuid
import pytest
from app.storage.graph import get_driver, setup_constraints, upsert_entity_node, upsert_evidence_node
from app.retrieval.graph_expansion import evidence_linked_to_entities, graph_expand

pytestmark = pytest.mark.asyncio

async def test_evidence_linked_to_entities_and_graph_expand():
    driver = get_driver()
    await setup_constraints(driver)
    entity_id = str(uuid.uuid4())
    evidence_id = str(uuid.uuid4())
    other_id = str(uuid.uuid4())
    async with driver.session() as session:
        await session.execute_write(upsert_entity_node, {
            "id": entity_id, "canonical_name": "Redis", "entity_type": "technology",
            "description": None, "aliases": [],
        })
        await session.execute_write(upsert_evidence_node, {
            "id": evidence_id, "content_text": "x", "modality": "speech", "source_id": "s",
            "timestamp_start": 0.0, "timestamp_end": 1.0, "confidence": 0.9,
        })
        await session.execute_write(upsert_evidence_node, {
            "id": other_id, "content_text": "y", "modality": "ocr", "source_id": "s",
            "timestamp_start": 2.0, "timestamp_end": 2.0, "confidence": 0.8,
        })
        await session.run(
            "MATCH (v:EvidenceUnit {id:$vid}), (e:Entity {id:$eid}) MERGE (v)-[:MENTIONS]->(e)",
            vid=evidence_id, eid=entity_id,
        )
        await session.run(
            "MATCH (a:EvidenceUnit {id:$a}), (b:EvidenceUnit {id:$b}) MERGE (a)-[:RELATES {relationship_type:'explains'}]->(b)",
            a=evidence_id, b=other_id,
        )

    linked = await evidence_linked_to_entities(driver, [entity_id])
    assert evidence_id in linked

    expanded = await graph_expand(driver, [evidence_id], hops=1)
    assert other_id in expanded
    await driver.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/retrieval/test_graph_expansion.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/retrieval/graph_expansion.py`**

```python
from app.extraction.ner import extract_entities
from app.linking.entity_resolution import resolve_entity

async def entities_for_query(session, query_text: str) -> list:
    mentions = extract_entities(query_text)
    return [await resolve_entity(session, m.text, m.entity_type) for m in mentions]

async def evidence_linked_to_entities(driver, entity_ids: list) -> list[str]:
    async with driver.session() as session:
        result = await session.run(
            "MATCH (v:EvidenceUnit)-[:MENTIONS]->(e:Entity) WHERE e.id IN $ids RETURN DISTINCT v.id AS id",
            ids=[str(i) for i in entity_ids],
        )
        return [record["id"] async for record in result]

async def graph_expand(driver, seed_ids: list[str], hops: int = 2) -> list[str]:
    async with driver.session() as session:
        result = await session.run(
            f"MATCH (start) WHERE start.id IN $ids "
            f"MATCH (start)-[*1..{hops}]-(connected) "
            "RETURN DISTINCT connected.id AS id",
            ids=[str(i) for i in seed_ids],
        )
        return [record["id"] async for record in result]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose up -d neo4j && cd backend && python -m pytest tests/retrieval/test_graph_expansion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/retrieval/graph_expansion.py backend/tests/retrieval/test_graph_expansion.py
git commit -m "feat: add entity-path retrieval and graph expansion"
```

---

### Task 28: Contradiction-aware fusion and rerank

**Files:**
- Create: `backend/app/retrieval/fusion_rerank.py`
- Test: `backend/tests/retrieval/test_fusion_rerank.py`

**Interfaces:**
- Produces: `RankedItem` dataclass (`id: str, score: float, superseded: bool = False`), `fuse_and_rerank(*, vector_hits: dict[str, float], graph_hit_ids: set[str], entity_hit_ids: set[str], confidences: dict[str, float], superseded_by: dict[str, str] | None = None) -> list[RankedItem]` in `backend/app/retrieval/fusion_rerank.py`. Score per id = `(0.5*vector_hits.get(id, 0.0) + 0.3*(1.0 if id in graph_hit_ids else 0.0) + 0.2*(1.0 if id in entity_hit_ids else 0.0)) * confidences.get(id, 1.0)`. Any id present as a key in `superseded_by` gets `superseded=True` and its score multiplied by `0.5` (still returned, so the UI can show "superseded" evidence, per spec §Retrieval step 4). Sorted descending by score.
- Consumes: nothing new — pure function over plain dicts/sets so it's trivially unit-testable without DB/graph fixtures.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/retrieval/test_fusion_rerank.py
from app.retrieval.fusion_rerank import fuse_and_rerank

def test_fuse_and_rerank_orders_by_weighted_score():
    ranked = fuse_and_rerank(
        vector_hits={"a": 0.9, "b": 0.3},
        graph_hit_ids={"b"},
        entity_hit_ids={"a", "b"},
        confidences={"a": 1.0, "b": 1.0},
    )
    ids = [r.id for r in ranked]
    assert set(ids) == {"a", "b"}
    assert ranked[0].score >= ranked[1].score

def test_fuse_and_rerank_downweights_superseded_evidence():
    ranked = fuse_and_rerank(
        vector_hits={"old": 0.9, "new": 0.9},
        graph_hit_ids=set(), entity_hit_ids=set(),
        confidences={"old": 1.0, "new": 1.0},
        superseded_by={"old": "new"},
    )
    old_item = next(r for r in ranked if r.id == "old")
    new_item = next(r for r in ranked if r.id == "new")
    assert old_item.superseded is True
    assert old_item.score < new_item.score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/retrieval/test_fusion_rerank.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/retrieval/fusion_rerank.py`**

```python
from dataclasses import dataclass

@dataclass
class RankedItem:
    id: str
    score: float
    superseded: bool = False

def fuse_and_rerank(*, vector_hits: dict, graph_hit_ids: set, entity_hit_ids: set,
                     confidences: dict, superseded_by: dict | None = None) -> list[RankedItem]:
    superseded_by = superseded_by or {}
    all_ids = set(vector_hits) | graph_hit_ids | entity_hit_ids
    items = []
    for item_id in all_ids:
        base = (0.5 * vector_hits.get(item_id, 0.0)
                + 0.3 * (1.0 if item_id in graph_hit_ids else 0.0)
                + 0.2 * (1.0 if item_id in entity_hit_ids else 0.0))
        score = base * confidences.get(item_id, 1.0)
        is_superseded = item_id in superseded_by
        if is_superseded:
            score *= 0.5
        items.append(RankedItem(id=item_id, score=score, superseded=is_superseded))
    items.sort(key=lambda r: r.score, reverse=True)
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/retrieval/test_fusion_rerank.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/retrieval/fusion_rerank.py backend/tests/retrieval/test_fusion_rerank.py
git commit -m "feat: add weighted fusion/rerank with contradiction downweighting"
```

### Task 29: Grounded answer synthesis (multimodal, cited)

**Files:**
- Create: `backend/app/retrieval/answer_synthesis.py`
- Test: `backend/tests/retrieval/test_answer_synthesis.py`

**Interfaces:**
- Produces: `Citation` dataclass (`evidence_id: str, modality: str, source: str, timestamp_start: float | None, timestamp_end: float | None, page_number: int | None, thumbnail: str | None, confidence: float`), `AnswerResult` dataclass (`answer: str, citations: list[Citation], events_used: list[str]`), `async def synthesize_answer(query: str, ranked_evidence: list, ranked_events: list) -> AnswerResult` in `backend/app/retrieval/answer_synthesis.py`. The LLM call only returns `{answer: str, cited_evidence_ids: list[str]}` (structured output) — `Citation` objects are built locally from the `EvidenceUnit` rows already in hand (`raw_locator` as `source`/`thumbnail`), never trusting the model to invent metadata.
- Consumes: `EvidenceUnit`, `KnowledgeEvent` (Task 2).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/retrieval/test_answer_synthesis.py
import uuid
import pytest
from unittest.mock import patch, MagicMock
from app.storage.models import EvidenceUnit, KnowledgeEvent
from app.retrieval.answer_synthesis import synthesize_answer, AnswerResult

pytestmark = pytest.mark.asyncio

def _evidence(modality, text, confidence=0.9):
    return EvidenceUnit(
        id=uuid.uuid4(), content_text=text, content_type="x", modality=modality,
        source_id=uuid.uuid4(), source_type="video", raw_locator="frame.jpg",
        extraction_method="x", timestamp_start=2.0, timestamp_end=2.0, confidence=confidence,
        entities=[], metadata_json={},
    )

async def test_synthesize_answer_builds_citations_from_evidence_not_model():
    speech = _evidence("speech", "We propose Redis to reduce database load")
    frame = _evidence("ocr", "REDIS ARCHITECTURE DIAGRAM")
    event = KnowledgeEvent(id=uuid.uuid4(), event_type="diagram_explanation", title="t", summary="s",
                            timestamp_start=0.0, timestamp_end=5.0, confidence=0.9, entities=[], metadata_json={})

    fake_message = MagicMock()
    fake_message.parsed = {
        "answer": "Redis was proposed to reduce database load, shown in the accompanying architecture diagram.",
        "cited_evidence_ids": [str(speech.id), str(frame.id)],
    }
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]

    with patch("app.retrieval.answer_synthesis._call_answer_model", return_value=fake_completion):
        result = await synthesize_answer(
            "What architecture was proposed to reduce database load?",
            ranked_evidence=[speech, frame], ranked_events=[event],
        )

    assert isinstance(result, AnswerResult)
    assert len(result.citations) == 2
    assert {c.evidence_id for c in result.citations} == {str(speech.id), str(frame.id)}
    assert result.citations[0].source == "frame.jpg" or result.citations[1].source == "frame.jpg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/retrieval/test_answer_synthesis.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/retrieval/answer_synthesis.py`**

```python
import os
import json
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class Citation:
    evidence_id: str
    modality: str
    source: str
    timestamp_start: float | None
    timestamp_end: float | None
    page_number: int | None
    thumbnail: str | None
    confidence: float

@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    events_used: list[str]

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_answer", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "cited_evidence_ids"],
            "additionalProperties": False,
        },
    },
}

def _call_answer_model(query: str, ranked_evidence: list, ranked_events: list):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    evidence_block = "\n".join(
        f"[{u.id}] ({u.modality}, t={u.timestamp_start}) {u.content_text}" for u in ranked_evidence
    )
    events_block = "\n".join(f"- {e.title}: {e.summary}" for e in ranked_events)
    prompt = (
        f"Question: {query}\n\nRelevant knowledge events:\n{events_block}\n\n"
        f"Evidence (cite by the bracketed id, only ids that genuinely support the answer):\n{evidence_block}"
    )
    return client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], response_format=_SCHEMA,
    )

async def synthesize_answer(query: str, ranked_evidence: list, ranked_events: list) -> AnswerResult:
    completion = _call_answer_model(query, ranked_evidence, ranked_events)
    parsed = completion.choices[0].message.parsed
    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    by_id = {str(u.id): u for u in ranked_evidence}
    citations = []
    for evidence_id in parsed["cited_evidence_ids"]:
        unit = by_id.get(evidence_id)
        if unit is None:
            continue
        citations.append(Citation(
            evidence_id=evidence_id, modality=unit.modality, source=unit.raw_locator,
            timestamp_start=unit.timestamp_start, timestamp_end=unit.timestamp_end,
            page_number=unit.page_number, thumbnail=unit.raw_locator, confidence=unit.confidence,
        ))
    return AnswerResult(
        answer=parsed["answer"], citations=citations,
        events_used=[str(e.id) for e in ranked_events],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/retrieval/test_answer_synthesis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/retrieval/answer_synthesis.py backend/tests/retrieval/test_answer_synthesis.py
git commit -m "feat: add grounded multimodal answer synthesis with source-verified citations"
```

### Task 30: Query API endpoint

**Files:**
- Create: `backend/app/retrieval/pipeline.py`, `backend/app/api/query.py`
- Modify: `backend/app/main.py` (mount query router)
- Test: `backend/tests/api/test_query_api.py`

**Interfaces:**
- Produces: `async def run_query(session, driver, query_text: str, top_k: int = 10) -> AnswerResult` in `backend/app/retrieval/pipeline.py`, composing Tasks 26-29 exactly per spec §Retrieval (embed query -> vector search evidence + events -> `entities_for_query` -> `evidence_linked_to_entities` -> `graph_expand` from top vector hits -> `fuse_and_rerank` -> fetch top-N full rows -> `synthesize_answer`). `POST /query` (body `{"query": str, "top_k": int = 10}`) returns `{"answer": str, "citations": [...], "events_used": [...]}` in `backend/app/api/query.py`, router mounted in `backend/app/main.py`.
- Consumes: everything from Tasks 26-29.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/api/test_query_api.py
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.answer_synthesis import AnswerResult, Citation

def test_query_endpoint_returns_answer_and_citations():
    client = TestClient(app)
    fake_result = AnswerResult(
        answer="Redis was proposed to reduce database load.",
        citations=[Citation("id1", "speech", "video.mp4#t=1,2", 1.0, 2.0, None, None, 0.9)],
        events_used=["event1"],
    )
    with patch("app.api.query.run_query", new=AsyncMock(return_value=fake_result)):
        resp = client.post("/query", json={"query": "What architecture reduces database load?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == fake_result.answer
    assert len(body["citations"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/api/test_query_api.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/retrieval/pipeline.py`**

```python
from app.extraction.embeddings import embed_texts
from app.retrieval.vector_search import vector_search_evidence, vector_search_events
from app.retrieval.graph_expansion import entities_for_query, evidence_linked_to_entities, graph_expand
from app.retrieval.fusion_rerank import fuse_and_rerank
from app.retrieval.answer_synthesis import synthesize_answer
from app.storage.repository import get_evidence_units_by_source  # noqa: F401 (re-exported for convenience)
from sqlalchemy import select
from app.storage.models import EvidenceUnit, KnowledgeEvent

async def run_query(session, driver, query_text: str, top_k: int = 10):
    query_embedding = embed_texts([query_text])[0]

    evidence_hits = await vector_search_evidence(session, query_embedding, top_k=top_k)
    event_hits = await vector_search_events(session, query_embedding, top_k=top_k)
    vector_hits = {str(u.id): sim for u, sim in evidence_hits}

    entities = await entities_for_query(session, query_text)
    entity_hit_ids = set(await evidence_linked_to_entities(driver, [e.id for e in entities])) if entities else set()

    seed_ids = list(vector_hits.keys())
    graph_hit_ids = set(await graph_expand(driver, seed_ids, hops=2)) if seed_ids else set()

    confidences = {str(u.id): u.confidence for u, _ in evidence_hits}
    ranked = fuse_and_rerank(
        vector_hits=vector_hits, graph_hit_ids=graph_hit_ids,
        entity_hit_ids=entity_hit_ids, confidences=confidences,
    )

    top_ids = [r.id for r in ranked[:top_k]]
    if top_ids:
        result = await session.execute(select(EvidenceUnit).where(EvidenceUnit.id.in_(top_ids)))
        by_id = {str(u.id): u for u in result.scalars().all()}
        ranked_evidence = [by_id[i] for i in top_ids if i in by_id]
    else:
        ranked_evidence = [u for u, _ in evidence_hits]

    ranked_events = [e for e, _ in event_hits]
    return await synthesize_answer(query_text, ranked_evidence, ranked_events)
```

- [ ] **Step 4: Implement `backend/app/api/query.py`**

```python
from fastapi import APIRouter
from pydantic import BaseModel
from app.storage.db import get_session
from app.storage.graph import get_driver
from app.retrieval.pipeline import run_query

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@router.post("/query")
async def query(request: QueryRequest):
    driver = get_driver()
    async with get_session() as session:
        result = await run_query(session, driver, request.query, request.top_k)
    await driver.close()
    return {
        "answer": result.answer,
        "citations": [c.__dict__ for c in result.citations],
        "events_used": result.events_used,
    }
```

- [ ] **Step 5: Mount the router in `backend/app/main.py`**

```python
from fastapi import FastAPI
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router

app = FastAPI(title="Multimodal RAG Pipeline")
app.include_router(ingest_router)
app.include_router(query_router)

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/api/test_query_api.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/retrieval/pipeline.py backend/app/api/query.py backend/app/main.py backend/tests/api/test_query_api.py
git commit -m "feat: add /query endpoint composing vector, graph, and entity retrieval paths"
```

---

## Phase 5 — Baseline text-only RAG (required comparison)

### Task 31: Naive fixed-chunk RAG baseline + /compare endpoint

**Files:**
- Modify: `backend/app/storage/repository.py` (add `persist_baseline_chunk`)
- Create: `backend/app/baseline/__init__.py`, `backend/app/baseline/naive_chunk_rag.py`, `backend/app/api/compare.py`
- Modify: `backend/app/ingestion/orchestrator.py` (also populate baseline chunks per ingested source)
- Modify: `backend/app/main.py` (mount compare router)
- Test: `backend/tests/baseline/test_naive_chunk_rag.py`, `backend/tests/api/test_compare_api.py`

**Interfaces:**
- Produces: `chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]` (word-based fixed chunking — deliberately naive, the one place in the codebase fixed-size chunking is allowed per Global Constraints), `async def ingest_baseline_from_source(session, source_id) -> list[BaselineChunk]` (pulls `EvidenceUnit`s for `source_id` with `content_type in ("transcript","ocr","document_text")`, concatenates in timestamp/page order, chunks, persists via `persist_baseline_chunk`), `async def baseline_vector_search(session, query_embedding, top_k=10) -> list[tuple[BaselineChunk, float]]`, `async def run_baseline_query(session, query_text: str, top_k: int = 10) -> str` (plain top-K vector search -> single LLM call -> plain string answer, no citations, no graph, no entities) — all in `backend/app/baseline/naive_chunk_rag.py`. `GET /compare?query=...` in `backend/app/api/compare.py` returns `{"multimodal": {...AnswerResult...}, "baseline": {"answer": str}}`.
- Consumes: `embed_texts` (Task 11), `get_evidence_units_by_source` (Task 12), `run_query` (Task 30).

- [ ] **Step 1: Write the failing test for chunking + baseline query**

```python
# backend/tests/baseline/test_naive_chunk_rag.py
import uuid
import pytest
from unittest.mock import patch, AsyncMock
from app.baseline.naive_chunk_rag import chunk_text, run_baseline_query

pytestmark = pytest.mark.asyncio

def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-50:] == second_words[:50]

async def test_run_baseline_query_does_plain_vector_search_and_llm_call():
    from app.storage.models import BaselineChunk
    chunk = BaselineChunk(id=uuid.uuid4(), content_text="Redis reduces load on Postgres.",
                           source_id=uuid.uuid4(), source_type="video", chunk_index=0, metadata_json={})
    with (
        patch("app.baseline.naive_chunk_rag.embed_texts", return_value=[[0.1] * 1536]),
        patch("app.baseline.naive_chunk_rag.baseline_vector_search", new=AsyncMock(return_value=[(chunk, 0.9)])),
        patch("app.baseline.naive_chunk_rag._call_baseline_answer_model") as mock_llm,
    ):
        mock_llm.return_value.choices = [type("M", (), {"message": type("M2", (), {"content": "Redis reduces database load."})()})]
        answer = await run_baseline_query(None, "What reduces database load?")
    assert "Redis" in answer
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/baseline/test_naive_chunk_rag.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Add `persist_baseline_chunk` to `backend/app/storage/repository.py`**

```python
async def persist_baseline_chunk(session, *, content_text, source_id, source_type, chunk_index) -> BaselineChunk:
    embedding = embed_texts([content_text])[0]
    chunk = BaselineChunk(
        id=uuid.uuid4(), content_text=content_text, source_id=source_id, source_type=source_type,
        chunk_index=chunk_index, embedding=embedding, metadata_json={},
    )
    session.add(chunk)
    await session.flush()
    return chunk
```

(add `BaselineChunk` to the existing import line from `app.storage.models` at the top of the file)

- [ ] **Step 4: Implement `backend/app/baseline/naive_chunk_rag.py`**

```python
import os
from openai import OpenAI
from sqlalchemy import select
from app.extraction.embeddings import embed_texts
from app.storage.repository import get_evidence_units_by_source, persist_baseline_chunk
from app.storage.models import BaselineChunk

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks

async def ingest_baseline_from_source(session, source_id) -> list[BaselineChunk]:
    units = await get_evidence_units_by_source(session, source_id)
    relevant = [u for u in units if u.content_type in ("transcript", "ocr", "document_text")]
    relevant.sort(key=lambda u: (u.page_number or 0, u.timestamp_start or 0.0))
    full_text = " ".join(u.content_text for u in relevant)
    chunks = chunk_text(full_text)
    return [
        await persist_baseline_chunk(session, content_text=c, source_id=source_id,
                                      source_type=relevant[0].source_type if relevant else "unknown",
                                      chunk_index=i)
        for i, c in enumerate(chunks)
    ]

async def baseline_vector_search(session, query_embedding, top_k: int = 10):
    distance = BaselineChunk.embedding.cosine_distance(query_embedding)
    result = await session.execute(
        select(BaselineChunk, distance.label("distance"))
        .where(BaselineChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    return [(row.BaselineChunk, 1.0 - row.distance) for row in result]

def _call_baseline_answer_model(query: str, chunks: list):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    context = "\n---\n".join(c.content_text for c in chunks)
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer using only the context."}],
    )

async def run_baseline_query(session, query_text: str, top_k: int = 10) -> str:
    query_embedding = embed_texts([query_text])[0]
    hits = await baseline_vector_search(session, query_embedding, top_k=top_k)
    chunks = [c for c, _ in hits]
    completion = _call_baseline_answer_model(query_text, chunks)
    return completion.choices[0].message.content
```

- [ ] **Step 5: Run the baseline test to verify it passes**

Run: `cd backend && python -m pytest tests/baseline/test_naive_chunk_rag.py -v`
Expected: PASS

- [ ] **Step 6: Wire baseline ingestion into `backend/app/ingestion/orchestrator.py`**

Add, right after the linking call inside `run_ingestion_job`:

```python
        from app.baseline.naive_chunk_rag import ingest_baseline_from_source
        async with get_session() as baseline_session:
            await ingest_baseline_from_source(baseline_session, source_id)
            await baseline_session.commit()
```

- [ ] **Step 7: Write the failing test for `/compare`**

```python
# backend/tests/api/test_compare_api.py
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.answer_synthesis import AnswerResult

def test_compare_endpoint_returns_both_systems_answers():
    client = TestClient(app)
    fake_multimodal = AnswerResult(answer="Redis, explained by the speaker over the architecture diagram.",
                                    citations=[], events_used=[])
    with (
        patch("app.api.compare.run_query", new=AsyncMock(return_value=fake_multimodal)),
        patch("app.api.compare.run_baseline_query", new=AsyncMock(return_value="Redis was mentioned.")),
    ):
        resp = client.get("/compare", params={"query": "What reduces database load?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "diagram" in body["multimodal"]["answer"]
    assert body["baseline"]["answer"] == "Redis was mentioned."
```

- [ ] **Step 8: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/api/test_compare_api.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 9: Implement `backend/app/api/compare.py`**

```python
from fastapi import APIRouter
from app.storage.db import get_session
from app.storage.graph import get_driver
from app.retrieval.pipeline import run_query
from app.baseline.naive_chunk_rag import run_baseline_query

router = APIRouter()

@router.get("/compare")
async def compare(query: str):
    driver = get_driver()
    async with get_session() as session:
        multimodal_result = await run_query(session, driver, query)
        baseline_answer = await run_baseline_query(session, query)
    await driver.close()
    return {
        "multimodal": {
            "answer": multimodal_result.answer,
            "citations": [c.__dict__ for c in multimodal_result.citations],
            "events_used": multimodal_result.events_used,
        },
        "baseline": {"answer": baseline_answer},
    }
```

- [ ] **Step 10: Mount the router in `backend/app/main.py`**

```python
from app.api.compare import router as compare_router
...
app.include_router(compare_router)
```

- [ ] **Step 11: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/api/test_compare_api.py -v`
Expected: PASS

- [ ] **Step 12: Commit**

```bash
git add backend/app/storage/repository.py backend/app/baseline backend/app/ingestion/orchestrator.py backend/app/api/compare.py backend/app/main.py backend/tests/baseline backend/tests/api/test_compare_api.py
git commit -m "feat: add naive fixed-chunk RAG baseline and /compare endpoint"
```

---

## Phase 6 — Frontend

"Futuristic 3D UI" is delivered via a dark, glass/neon-accented layout (Tailwind) plus a genuinely 3D force-directed knowledge-graph view (`3d-force-graph`, which is three.js under the hood) — not a decorative skin. Component tests use Vitest + React Testing Library; the 3D view gets a smoke test with the graph library mocked (a real WebGL context isn't available in the test runner).

### Task 32: Frontend scaffold + typed API client

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tsconfig.node.json`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/index.css`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/src/api/client.ts`, `frontend/Dockerfile`
- Test: `frontend/src/api/client.test.ts`

**Interfaces:**
- Produces: types `JobStatus`, `Citation`, `QueryResponse`, `CompareResponse` and `apiClient = { ingest, getIngestStatus, query, compare }` in `frontend/src/api/client.ts` — every later component imports these exact names/shapes.
- Consumes: nothing (foundational).

- [ ] **Step 1: Write `frontend/package.json`**

```json
{
  "name": "multimodal-rag-frontend",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "test": "vitest run"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "3d-force-graph": "^1.73.0",
    "three": "^0.170.0"
  },
  "devDependencies": {
    "@testing-library/react": "^16.0.1",
    "@testing-library/jest-dom": "^6.6.3",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.3",
    "autoprefixer": "^10.4.20",
    "jsdom": "^25.0.1",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.14",
    "typescript": "^5.6.3",
    "vite": "^5.4.10",
    "vitest": "^2.1.4"
  }
}
```

- [ ] **Step 2: Write `frontend/vite.config.ts`**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173 },
  test: { environment: "jsdom", globals: true },
});
```

- [ ] **Step 3: Write `frontend/tsconfig.json`, `frontend/tsconfig.node.json`, `frontend/index.html`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`**

```json
// frontend/tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020", "useDefineForClassFields": true, "lib": ["ES2020", "DOM"],
    "module": "ESNext", "skipLibCheck": true, "moduleResolution": "bundler",
    "resolveJsonModule": true, "isolatedModules": true, "noEmit": true, "jsx": "react-jsx", "strict": true
  },
  "include": ["src"]
}
```

```json
// frontend/tsconfig.node.json
{ "compilerOptions": { "composite": true, "module": "ESNext", "moduleResolution": "bundler" }, "include": ["vite.config.ts"] }
```

```html
<!-- frontend/index.html -->
<!doctype html>
<html lang="en">
  <head><meta charset="UTF-8" /><title>Multimodal RAG Pipeline</title></head>
  <body class="bg-slate-950 text-slate-100">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

```javascript
// frontend/tailwind.config.js
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: { colors: { accent: "#22d3ee" } } },
  plugins: [],
};
```

```javascript
// frontend/postcss.config.js
export default { plugins: { tailwindcss: {}, autoprefixer: {} } };
```

- [ ] **Step 4: Write `frontend/src/index.css`, `frontend/src/main.tsx`**

```css
/* frontend/src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;
```

```tsx
// frontend/src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode><App /></React.StrictMode>
);
```

- [ ] **Step 5: Write the failing test for the API client**

```typescript
// frontend/src/api/client.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { apiClient } from "./client";

describe("apiClient", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  it("query posts to /query and returns parsed JSON", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ answer: "Redis", citations: [], events_used: [] }),
    });
    const result = await apiClient.query("What reduces load?");
    expect(result.answer).toBe("Redis");
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/query"),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("getIngestStatus throws on a non-ok response", async () => {
    (global.fetch as any).mockResolvedValue({ ok: false, status: 404 });
    await expect(apiClient.getIngestStatus("bad-id")).rejects.toThrow();
  });
});
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd frontend && npm install && npm test`
Expected: FAIL (Cannot find module './client')

- [ ] **Step 7: Implement `frontend/src/api/client.ts`**

```typescript
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "done" | "failed";
  source_id: string | null;
  error: string | null;
}

export interface Citation {
  evidence_id: string;
  modality: string;
  source: string;
  timestamp_start: number | null;
  timestamp_end: number | null;
  page_number: number | null;
  thumbnail: string | null;
  confidence: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  events_used: string[];
}

export interface CompareResponse {
  multimodal: QueryResponse;
  baseline: { answer: string };
}

async function ingest(file: File, sourceType: string): Promise<{ job_id: string }> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("source_type", sourceType);
  const res = await fetch(`${BASE_URL}/ingest`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(`ingest failed: ${res.status}`);
  return res.json();
}

async function getIngestStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE_URL}/ingest/${jobId}`);
  if (!res.ok) throw new Error(`status check failed: ${res.status}`);
  return res.json();
}

async function query(queryText: string, topK = 10): Promise<QueryResponse> {
  const res = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: queryText, top_k: topK }),
  });
  if (!res.ok) throw new Error(`query failed: ${res.status}`);
  return res.json();
}

async function compare(queryText: string): Promise<CompareResponse> {
  const res = await fetch(`${BASE_URL}/compare?query=${encodeURIComponent(queryText)}`);
  if (!res.ok) throw new Error(`compare failed: ${res.status}`);
  return res.json();
}

export const apiClient = { ingest, getIngestStatus, query, compare };
```

- [ ] **Step 8: Write a minimal `frontend/src/App.tsx` placeholder (replaced by Task 33-37)**

```tsx
export default function App() {
  return <div className="min-h-screen p-8">Multimodal RAG Pipeline</div>;
}
```

- [ ] **Step 9: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 10: Write `frontend/Dockerfile`**

```dockerfile
FROM node:20-slim
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
CMD ["npm", "run", "dev", "--", "--host"]
```

- [ ] **Step 11: Commit**

```bash
git add frontend/package.json frontend/vite.config.ts frontend/tsconfig*.json frontend/index.html \
        frontend/tailwind.config.js frontend/postcss.config.js frontend/src/index.css frontend/src/main.tsx \
        frontend/src/App.tsx frontend/src/api/client.ts frontend/src/api/client.test.ts frontend/Dockerfile
git commit -m "chore: scaffold Vite/React/TS/Tailwind frontend with typed API client"
```

### Task 33: QueryPanel + EvidenceCard components

**Files:**
- Modify: `frontend/vite.config.ts` (add `test.setupFiles`)
- Create: `frontend/src/setupTests.ts`, `frontend/src/components/EvidenceCard.tsx`, `frontend/src/components/QueryPanel.tsx`
- Test: `frontend/src/components/QueryPanel.test.tsx`

**Interfaces:**
- Produces: `EvidenceCard({ citation: Citation })` and `QueryPanel()` (no props) in `frontend/src/components/`, both default/named exports as shown below — `QueryPanel` is what Task 37 mounts on the "query" tab.
- Consumes: `apiClient`, `Citation`, `QueryResponse` (Task 32).

- [ ] **Step 1: Add jest-dom setup**

`frontend/src/setupTests.ts`:
```typescript
import "@testing-library/jest-dom";
```

Update `frontend/vite.config.ts`'s `test` block: `test: { environment: "jsdom", globals: true, setupFiles: "./src/setupTests.ts" }`.

- [ ] **Step 2: Write the failing test**

```tsx
// frontend/src/components/QueryPanel.test.tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { QueryPanel } from "./QueryPanel";
import { apiClient } from "../api/client";

vi.mock("../api/client", () => ({ apiClient: { query: vi.fn() } }));

describe("QueryPanel", () => {
  it("submits a query and renders the answer and evidence cards", async () => {
    (apiClient.query as any).mockResolvedValue({
      answer: "Redis reduces database load.",
      citations: [{
        evidence_id: "e1", modality: "speech", source: "video.mp4#t=1,2",
        timestamp_start: 1, timestamp_end: 2, page_number: null, thumbnail: null, confidence: 0.9,
      }],
      events_used: ["ev1"],
    });
    render(<QueryPanel />);
    fireEvent.change(screen.getByPlaceholderText(/ask a question/i), { target: { value: "What reduces load?" } });
    fireEvent.click(screen.getByText("Search"));
    await waitFor(() => expect(screen.getByTestId("answer-text")).toHaveTextContent("Redis reduces database load."));
    expect(screen.getByText(/video.mp4/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL (Cannot find module './QueryPanel')

- [ ] **Step 4: Implement `frontend/src/components/EvidenceCard.tsx`**

```tsx
import { Citation } from "../api/client";

const MODALITY_LABEL: Record<string, string> = {
  speech: "Speech", ocr: "On-screen text", image: "Visual", document: "Document",
};

export function EvidenceCard({ citation }: { citation: Citation }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 backdrop-blur">
      <div className="flex items-center justify-between text-xs uppercase tracking-wide text-accent">
        <span>{MODALITY_LABEL[citation.modality] ?? citation.modality}</span>
        <span>{Math.round(citation.confidence * 100)}% confidence</span>
      </div>
      <p className="mt-2 text-sm text-slate-300 break-all">{citation.source}</p>
      {citation.timestamp_start !== null && (
        <p className="mt-1 text-xs text-slate-500">
          t = {citation.timestamp_start.toFixed(1)}s
          {citation.timestamp_end !== null && citation.timestamp_end !== citation.timestamp_start
            ? ` - ${citation.timestamp_end.toFixed(1)}s` : ""}
        </p>
      )}
      {citation.page_number !== null && <p className="mt-1 text-xs text-slate-500">page {citation.page_number}</p>}
    </div>
  );
}
```

- [ ] **Step 5: Implement `frontend/src/components/QueryPanel.tsx`**

```tsx
import { useState } from "react";
import { apiClient, QueryResponse } from "../api/client";
import { EvidenceCard } from "./EvidenceCard";

export function QueryPanel() {
  const [queryText, setQueryText] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResult(await apiClient.query(queryText));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          className="flex-1 rounded-md bg-slate-900 border border-slate-700 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-accent"
          placeholder="Ask a question across video, audio, images, and docs..."
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
        />
        <button type="submit" disabled={loading || !queryText}
                className="rounded-md bg-accent px-4 py-2 font-medium text-slate-950 disabled:opacity-50">
          {loading ? "Searching..." : "Search"}
        </button>
      </form>
      {error && <p role="alert" className="text-red-400">{error}</p>}
      {result && (
        <div className="flex flex-col gap-4">
          <p className="text-lg" data-testid="answer-text">{result.answer}</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {result.citations.map((citation) => <EvidenceCard key={citation.evidence_id} citation={citation} />)}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/vite.config.ts frontend/src/setupTests.ts frontend/src/components/EvidenceCard.tsx frontend/src/components/QueryPanel.tsx frontend/src/components/QueryPanel.test.tsx
git commit -m "feat: add QueryPanel and EvidenceCard components"
```

### Task 34: VideoPlayer (timestamp seek) + PDFViewer (page jump)

**Files:**
- Create: `frontend/src/components/VideoPlayer.tsx`, `frontend/src/components/PDFViewer.tsx`
- Test: `frontend/src/components/VideoPlayer.test.tsx`, `frontend/src/components/PDFViewer.test.tsx`

**Interfaces:**
- Produces: `VideoPlayer({ src: string, seekToSeconds?: number })` and `PDFViewer({ src: string, pageNumber?: number })`, both in `frontend/src/components/`. `PDFViewer` uses the browser's native PDF renderer via an iframe with a `#page=` fragment rather than a full `pdf.js` integration — a deliberate scope cut (still gives exact page-jump provenance, at a fraction of the integration cost) noted here for the README's "design decisions" section.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/components/VideoPlayer.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { VideoPlayer } from "./VideoPlayer";

describe("VideoPlayer", () => {
  it("renders with the given source", () => {
    render(<VideoPlayer src="video.mp4" />);
    const video = screen.getByTestId("video-player") as HTMLVideoElement;
    expect(video.src).toContain("video.mp4");
  });

  it("seeks to the given timestamp when seekToSeconds is provided", () => {
    render(<VideoPlayer src="video.mp4" seekToSeconds={5} />);
    const video = screen.getByTestId("video-player") as HTMLVideoElement;
    expect(video.currentTime).toBe(5);
  });
});
```

```tsx
// frontend/src/components/PDFViewer.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PDFViewer } from "./PDFViewer";

describe("PDFViewer", () => {
  it("includes the page number as a URL fragment", () => {
    render(<PDFViewer src="doc.pdf" pageNumber={3} />);
    const iframe = screen.getByTestId("pdf-viewer") as HTMLIFrameElement;
    expect(iframe.src).toContain("doc.pdf#page=3");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test`
Expected: FAIL (Cannot find module)

- [ ] **Step 3: Implement `frontend/src/components/VideoPlayer.tsx`**

```tsx
import { useEffect, useRef } from "react";

export function VideoPlayer({ src, seekToSeconds }: { src: string; seekToSeconds?: number }) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && seekToSeconds !== undefined) {
      videoRef.current.currentTime = seekToSeconds;
    }
  }, [seekToSeconds]);

  return <video ref={videoRef} src={src} controls className="w-full rounded-lg" data-testid="video-player" />;
}
```

- [ ] **Step 4: Implement `frontend/src/components/PDFViewer.tsx`**

```tsx
export function PDFViewer({ src, pageNumber }: { src: string; pageNumber?: number }) {
  const hash = pageNumber ? `#page=${pageNumber}` : "";
  return (
    <iframe src={`${src}${hash}`} title="PDF preview"
            className="w-full h-[600px] rounded-lg border border-slate-800" data-testid="pdf-viewer" />
  );
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/VideoPlayer.tsx frontend/src/components/PDFViewer.tsx frontend/src/components/VideoPlayer.test.tsx frontend/src/components/PDFViewer.test.tsx
git commit -m "feat: add VideoPlayer with timestamp seek and PDFViewer with page jump"
```

### Task 35: Graph API endpoint + 3D GraphView component

**Files:**
- Create: `backend/app/api/graph.py`
- Modify: `backend/app/main.py` (mount graph router)
- Test: `backend/tests/api/test_graph_api.py`
- Create: `frontend/src/components/GraphView.tsx`
- Test: `frontend/src/components/GraphView.test.tsx`

**Interfaces:**
- Produces: `GET /graph` returns `{"nodes": [{"id": str, "label": str, "type": "entity"|"evidence"|"event"}], "links": [{"source": str, "target": str, "relationship_type": str}]}` in `backend/app/api/graph.py`, querying Neo4j directly (capped at 500 nodes for demo responsiveness). `GraphView({ nodes: GraphNode[], links: GraphLink[] })` in `frontend/src/components/GraphView.tsx`, wrapping `3d-force-graph`.
- Consumes: `get_driver` (Task 3).

- [ ] **Step 1: Write the failing backend test**

```python
# backend/tests/api/test_graph_api.py
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

def test_graph_endpoint_returns_nodes_and_links():
    client = TestClient(app)
    with patch("app.api.graph.fetch_graph_snapshot", new=AsyncMock(return_value={
        "nodes": [{"id": "1", "label": "Redis", "type": "entity"}],
        "links": [{"source": "1", "target": "2", "relationship_type": "explains"}],
    })):
        resp = client.get("/graph")
    assert resp.status_code == 200
    assert resp.json()["nodes"][0]["label"] == "Redis"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/api/test_graph_api.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `backend/app/api/graph.py`**

```python
from fastapi import APIRouter
from app.storage.graph import get_driver

router = APIRouter()

async def fetch_graph_snapshot(limit: int = 500) -> dict:
    driver = get_driver()
    async with driver.session() as session:
        node_result = await session.run(
            "MATCH (n) WHERE n:Entity OR n:EvidenceUnit OR n:KnowledgeEvent "
            "RETURN n.id AS id, coalesce(n.canonical_name, n.title, left(n.content_text, 60)) AS label, "
            "CASE WHEN n:Entity THEN 'entity' WHEN n:KnowledgeEvent THEN 'event' ELSE 'evidence' END AS type "
            "LIMIT $limit",
            limit=limit,
        )
        nodes = [dict(record) async for record in node_result]
        link_result = await session.run(
            "MATCH (a)-[r]->(b) WHERE a.id IS NOT NULL AND b.id IS NOT NULL "
            "RETURN a.id AS source, b.id AS target, coalesce(r.relationship_type, type(r)) AS relationship_type "
            "LIMIT $limit",
            limit=limit,
        )
        links = [dict(record) async for record in link_result]
    await driver.close()
    return {"nodes": nodes, "links": links}

@router.get("/graph")
async def graph():
    return await fetch_graph_snapshot()
```

- [ ] **Step 4: Mount the router in `backend/app/main.py`**

```python
from app.api.graph import router as graph_router
...
app.include_router(graph_router)
```

- [ ] **Step 5: Run backend test to verify it passes**

Run: `cd backend && python -m pytest tests/api/test_graph_api.py -v`
Expected: PASS

- [ ] **Step 6: Write the failing frontend test**

```tsx
// frontend/src/components/GraphView.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { GraphView } from "./GraphView";

vi.mock("3d-force-graph", () => ({
  default: () => {
    const chain: any = {
      graphData: () => chain, nodeAutoColorBy: () => chain, nodeLabel: () => chain,
      linkDirectionalArrowLength: () => chain, linkLabel: () => chain,
    };
    return () => chain;
  },
}));

describe("GraphView", () => {
  it("renders a container for the 3d force graph without crashing", () => {
    render(<GraphView nodes={[{ id: "1", label: "Redis", type: "entity" }]} links={[]} />);
    expect(screen.getByTestId("graph-view")).toBeInTheDocument();
  });
});
```

- [ ] **Step 7: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL (Cannot find module './GraphView')

- [ ] **Step 8: Implement `frontend/src/components/GraphView.tsx`**

```tsx
import { useEffect, useRef } from "react";
import ForceGraph3D from "3d-force-graph";

export interface GraphNode { id: string; label: string; type: "entity" | "evidence" | "event"; }
export interface GraphLink { source: string; target: string; relationship_type: string; }

export function GraphView({ nodes, links }: { nodes: GraphNode[]; links: GraphLink[] }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    ForceGraph3D()(containerRef.current)
      .graphData({ nodes, links })
      .nodeAutoColorBy("type")
      .nodeLabel((n: any) => n.label)
      .linkDirectionalArrowLength(4)
      .linkLabel((l: any) => l.relationship_type);
  }, [nodes, links]);

  return <div ref={containerRef} data-testid="graph-view" className="w-full h-[500px] rounded-lg border border-slate-800" />;
}
```

- [ ] **Step 9: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add backend/app/api/graph.py backend/app/main.py backend/tests/api/test_graph_api.py frontend/src/components/GraphView.tsx frontend/src/components/GraphView.test.tsx
git commit -m "feat: add /graph endpoint and 3D force-directed GraphView"
```

### Task 36: CompareView (multimodal vs. baseline side-by-side)

**Files:**
- Create: `frontend/src/components/CompareView.tsx`
- Test: `frontend/src/components/CompareView.test.tsx`

**Interfaces:**
- Produces: `CompareView()` (no props) in `frontend/src/components/CompareView.tsx`.
- Consumes: `apiClient.compare`, `CompareResponse` (Task 32).

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/CompareView.test.tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { CompareView } from "./CompareView";
import { apiClient } from "../api/client";

vi.mock("../api/client", () => ({ apiClient: { compare: vi.fn() } }));

describe("CompareView", () => {
  it("shows both systems' answers after comparing", async () => {
    (apiClient.compare as any).mockResolvedValue({
      multimodal: { answer: "Redis, explained over the architecture diagram.", citations: [{}], events_used: ["e1"] },
      baseline: { answer: "Redis was mentioned." },
    });
    render(<CompareView />);
    fireEvent.change(screen.getByPlaceholderText(/compare a query/i), { target: { value: "What reduces load?" } });
    fireEvent.click(screen.getByText("Compare"));
    await waitFor(() => expect(screen.getByTestId("multimodal-answer")).toHaveTextContent("architecture diagram"));
    expect(screen.getByTestId("baseline-answer")).toHaveTextContent("Redis was mentioned.");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL (Cannot find module './CompareView')

- [ ] **Step 3: Implement `frontend/src/components/CompareView.tsx`**

```tsx
import { useState } from "react";
import { apiClient, CompareResponse } from "../api/client";

export function CompareView() {
  const [queryText, setQueryText] = useState("");
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      setResult(await apiClient.compare(queryText));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          className="flex-1 rounded-md bg-slate-900 border border-slate-700 px-4 py-2"
          placeholder="Compare a query across both systems..."
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
        />
        <button type="submit" disabled={loading || !queryText} className="rounded-md bg-accent px-4 py-2 text-slate-950">
          {loading ? "Comparing..." : "Compare"}
        </button>
      </form>
      {result && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-lg border border-accent/50 bg-slate-900/60 p-4">
            <h3 className="text-accent font-semibold mb-2">Multimodal system</h3>
            <p data-testid="multimodal-answer">{result.multimodal.answer}</p>
            <p className="mt-2 text-xs text-slate-500">{result.multimodal.citations.length} cross-modal citations</p>
          </div>
          <div className="rounded-lg border border-slate-700 bg-slate-900/60 p-4">
            <h3 className="text-slate-400 font-semibold mb-2">Text-only baseline</h3>
            <p data-testid="baseline-answer">{result.baseline.answer}</p>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/CompareView.tsx frontend/src/components/CompareView.test.tsx
git commit -m "feat: add CompareView showing multimodal vs. baseline answers side by side"
```

### Task 37: UploadPanel + final App.tsx wiring (tabs)

**Files:**
- Create: `frontend/src/components/UploadPanel.tsx`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/components/UploadPanel.test.tsx`

**Interfaces:**
- Produces: `UploadPanel()` (no props) in `frontend/src/components/UploadPanel.tsx`. `App()` in `frontend/src/App.tsx` becomes the tabbed shell mounting `QueryPanel`, `CompareView`, `GraphView` (fetching `/graph` on mount), and `UploadPanel`.
- Consumes: `apiClient.ingest`, `apiClient.getIngestStatus` (Task 32); `QueryPanel` (Task 33); `CompareView` (Task 36); `GraphView` (Task 35).

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/UploadPanel.test.tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { UploadPanel } from "./UploadPanel";
import { apiClient } from "../api/client";

vi.mock("../api/client", () => ({ apiClient: { ingest: vi.fn(), getIngestStatus: vi.fn() } }));

describe("UploadPanel", () => {
  beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
  afterEach(() => vi.useRealTimers());

  it("uploads a file and polls status until done", async () => {
    (apiClient.ingest as any).mockResolvedValue({ job_id: "job1" });
    (apiClient.getIngestStatus as any).mockResolvedValue({ job_id: "job1", status: "done", source_id: "s1", error: null });

    render(<UploadPanel />);
    const file = new File(["data"], "video.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
    fireEvent.submit(screen.getByTestId("file-input").closest("form")!);

    await vi.advanceTimersByTimeAsync(2000);
    await waitFor(() => expect(screen.getByTestId("ingest-status")).toHaveTextContent("done"));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL (Cannot find module './UploadPanel')

- [ ] **Step 3: Implement `frontend/src/components/UploadPanel.tsx`**

```tsx
import { useRef, useState } from "react";
import { apiClient, JobStatus } from "../api/client";

const SOURCE_TYPES = ["video", "audio", "image", "pdf"];

export function UploadPanel() {
  const [sourceType, setSourceType] = useState("video");
  const [status, setStatus] = useState<JobStatus | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;
    const { job_id } = await apiClient.ingest(file, sourceType);
    poll(job_id);
  }

  function poll(jobId: string) {
    const interval = setInterval(async () => {
      const s = await apiClient.getIngestStatus(jobId);
      setStatus(s);
      if (s.status === "done" || s.status === "failed") clearInterval(interval);
    }, 2000);
  }

  return (
    <form onSubmit={handleUpload} className="flex flex-col gap-3">
      <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}
              className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 w-fit">
        {SOURCE_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
      </select>
      <input ref={fileInputRef} type="file" data-testid="file-input" className="text-sm" />
      <button type="submit" className="rounded-md bg-accent px-4 py-2 text-slate-950 w-fit">Ingest</button>
      {status && <p data-testid="ingest-status">Status: {status.status}</p>}
    </form>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 5: Implement final `frontend/src/App.tsx`**

```tsx
import { useEffect, useState } from "react";
import { QueryPanel } from "./components/QueryPanel";
import { CompareView } from "./components/CompareView";
import { GraphView, GraphNode, GraphLink } from "./components/GraphView";
import { UploadPanel } from "./components/UploadPanel";

type Tab = "query" | "compare" | "graph" | "upload";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function App() {
  const [tab, setTab] = useState<Tab>("query");
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({ nodes: [], links: [] });

  useEffect(() => {
    if (tab !== "graph") return;
    fetch(`${BASE_URL}/graph`).then((r) => r.json()).then(setGraphData).catch(() => {});
  }, [tab]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-8">
      <h1 className="text-2xl font-bold text-accent mb-6">Multimodal RAG Pipeline</h1>
      <nav className="flex gap-4 mb-6 text-sm">
        {(["query", "compare", "graph", "upload"] as Tab[]).map((t) => (
          <button key={t} onClick={() => setTab(t)}
                  className={`px-3 py-1 rounded-md capitalize ${tab === t ? "bg-accent text-slate-950" : "bg-slate-800"}`}>
            {t}
          </button>
        ))}
      </nav>
      {tab === "query" && <QueryPanel />}
      {tab === "compare" && <CompareView />}
      {tab === "graph" && <GraphView nodes={graphData.nodes} links={graphData.links} />}
      {tab === "upload" && <UploadPanel />}
    </div>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/UploadPanel.tsx frontend/src/components/UploadPanel.test.tsx frontend/src/App.tsx
git commit -m "feat: add UploadPanel and wire tabbed App shell (query/compare/graph/upload)"
```

---

## Phase 7 — Evaluation against the baseline

### Task 38: Eval harness (multimodal vs. baseline, modality-recall report)

**Files:**
- Create: `eval/queries.json`, `eval/run_eval.py`, `eval/test_run_eval.py`

**Interfaces:**
- Produces: `compute_modality_recall(required_modalities: set, retrieved_modalities: set) -> float` (fraction of `required_modalities` present in `retrieved_modalities`; `1.0` if `required_modalities` is empty), `format_report(results: list[dict]) -> str` (markdown report with per-query recall/answers and an average), and `run(base_url: str, queries_path: str, report_path: str) -> None` (hits the live `/query` and `/compare` endpoints for every entry in `queries.json`) — all in `eval/run_eval.py`.
- Consumes: the running backend's `/query` and `/compare` endpoints (Tasks 30-31) — this is the one integration point in the plan that talks to a fully running system rather than mocks, by design (it's the deliverable comparison).

`eval/queries.json` ships with the schema and one illustrative placeholder entry — the real entries get filled in against whatever real video/audio/PDF the demo dataset turns out to be (per the "no hardcoded demo content" constraint, gold queries can't be authored before that data exists).

- [ ] **Step 1: Write `eval/queries.json`**

```json
[
  {
    "id": "q1",
    "query": "REPLACE ME: a question whose correct answer requires connecting speech and an on-screen diagram",
    "required_modalities": ["speech", "ocr"]
  }
]
```

- [ ] **Step 2: Write the failing test for the pure scoring/formatting functions**

```python
# eval/test_run_eval.py
from run_eval import compute_modality_recall, format_report

def test_compute_modality_recall_full_match():
    assert compute_modality_recall({"speech", "ocr"}, {"speech", "ocr", "image"}) == 1.0

def test_compute_modality_recall_partial_match():
    assert compute_modality_recall({"speech", "ocr"}, {"speech"}) == 0.5

def test_compute_modality_recall_empty_requirement_is_perfect():
    assert compute_modality_recall(set(), {"speech"}) == 1.0

def test_format_report_includes_average_and_query_id():
    report = format_report([
        {"id": "q1", "query": "x", "multimodal_recall": 1.0, "multimodal_answer": "a", "baseline_answer": "b"},
    ])
    assert "Average modality recall" in report
    assert "q1" in report
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd eval && python -m pytest test_run_eval.py -v`
Expected: FAIL (ModuleNotFoundError: run_eval)

- [ ] **Step 4: Implement `eval/run_eval.py`**

```python
import json
import sys
import httpx

def compute_modality_recall(required_modalities: set, retrieved_modalities: set) -> float:
    if not required_modalities:
        return 1.0
    return len(required_modalities & retrieved_modalities) / len(required_modalities)

def format_report(results: list) -> str:
    avg = sum(r["multimodal_recall"] for r in results) / len(results) if results else 0.0
    lines = ["# Evaluation Report", "", f"Average modality recall across {len(results)} queries: {avg:.2f}", ""]
    for r in results:
        lines.append(f"## {r['id']}: {r['query']}")
        lines.append(f"- Multimodal modality recall: {r['multimodal_recall']:.2f}")
        lines.append(f"- Multimodal answer: {r['multimodal_answer']}")
        lines.append(f"- Baseline answer: {r['baseline_answer']}")
        lines.append("")
    return "\n".join(lines)

def run(base_url: str, queries_path: str, report_path: str) -> None:
    with open(queries_path) as f:
        queries = json.load(f)
    results = []
    with httpx.Client(timeout=120) as client:
        for q in queries:
            multimodal = client.post(f"{base_url}/query", json={"query": q["query"]}).json()
            compared = client.get(f"{base_url}/compare", params={"query": q["query"]}).json()
            retrieved_modalities = {c["modality"] for c in multimodal["citations"]}
            recall = compute_modality_recall(set(q["required_modalities"]), retrieved_modalities)
            results.append({
                "id": q["id"], "query": q["query"], "multimodal_recall": recall,
                "multimodal_answer": multimodal["answer"], "baseline_answer": compared["baseline"]["answer"],
            })
    report = format_report(results)
    with open(report_path, "w") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000", "eval/queries.json", "eval/report.md")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd eval && python -m pytest test_run_eval.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add eval/queries.json eval/run_eval.py eval/test_run_eval.py
git commit -m "feat: add eval harness comparing multimodal and baseline systems by modality recall"
```

---

## Phase 8 — README and end-to-end validation

### Task 39: README, architecture write-up, and a full docker-compose smoke test

**Files:**
- Modify: `README.md`

**Interfaces:** none (documentation + manual verification task).

- [ ] **Step 1: Bring the whole stack up**

Run: `docker compose up -d --build` then `docker compose exec backend alembic upgrade head`
Expected: all four services (`postgres`, `neo4j`, `backend`, `frontend`) report healthy/running; `curl http://localhost:8000/health` returns `{"status":"ok"}`; `http://localhost:5173` loads the frontend shell.

- [ ] **Step 2: Ingest one real file of each of the three required modalities and confirm linking runs**

Through the Upload tab (or `curl -F file=@... -F source_type=video http://localhost:8000/ingest`), ingest one real video, one real PDF, and either a real audio clip or image, then poll `/ingest/{job_id}` until `"status": "done"` for each. Confirm via `docker compose exec postgres psql -U ragdb -c 'select modality, count(*) from evidence_units group by modality;'` that units of more than one modality exist, and `select count(*) from knowledge_events;` is non-zero.

- [ ] **Step 3: Run one real cross-modal query through the UI and through `/compare`**

Ask a question in the Query tab whose answer requires connecting speech to a frame or a document, and via `curl http://localhost:8000/compare?query=...` confirm the multimodal answer cites more than one modality while noting how the baseline answer compares.

- [ ] **Step 4: Fill in `eval/queries.json` against the real ingested data and run the eval harness**

Run: `cd eval && python run_eval.py http://localhost:8000` and confirm `eval/report.md` is generated with a non-trivial average modality recall.

- [ ] **Step 5: Write `README.md`**

```markdown
# Multimodal Data Management Pipeline for RAG-Ready Systems

## What this is
[1-2 sentences restating the problem being solved, in this project's own words]

## Quickstart
1. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` (and `HF_TOKEN` for pyannote diarization — free at huggingface.co, accept the `pyannote/speaker-diarization-3.1` model terms).
2. `docker compose up -d --build`
3. `docker compose exec backend alembic upgrade head`
4. Open `http://localhost:5173`, go to the Upload tab, ingest a video/audio/image/PDF.
5. Once ingestion shows `done`, ask a question in the Query tab.

## Architecture
[Paste/adapt the architecture section from docs/superpowers/specs/2026-08-22-multimodal-rag-pipeline-design.md:
 the four-object data model, the segmentation -> candidate-scoring -> relationship-inference ->
 entity-resolution -> event-assembly -> contradiction pipeline, and the hybrid retrieval flow.]

## Key design decisions
- Four first-class objects (EvidenceUnit/KnowledgeEvent/Entity/Relationship) instead of one generic
  "chunk," so cross-modal structure survives into storage instead of being flattened away.
- Cheap scoring (timestamp/keyword/embedding) before any LLM relationship call, bounding LLM cost
  regardless of video length.
- Every Relationship stores its component scores, not just a bare confidence number — auditable.
- A naive fixed-chunk baseline is a first-class sibling system, not an afterthought, so the
  comparison in `eval/report.md` is real.

## Evaluation
See `eval/report.md` for the multimodal-vs-baseline comparison on real ingested data.

## Future improvements
- True `pdf.js` integration (currently a native-iframe page-jump) for in-page bbox highlighting.
- Cross-file entity/event linking surfaced explicitly in the graph view (currently implicit via
  shared Entity nodes).
- Streaming ingestion progress instead of polling.
```

- [ ] **Step 6: Commit**

```bash
git add README.md eval/report.md
git commit -m "docs: add README with architecture, design decisions, and future improvements"
```

---

## Plan self-review notes

- **Spec coverage:** Multimodal ingestion (Tasks 4-17) — video/audio/image/PDF ✓. Structured representation (Tasks 2, 12) — all eight required fields per object ✓. Cross-modal/temporal relationships (Tasks 18-25) ✓. Retrieval + traceability (Tasks 26-30, 33-34) ✓. Baseline comparison deliverable (Task 31, 38) ✓. Confidence scores used at extraction (Tasks 5, 7-9) and at retrieval (Task 28) ✓. Contradiction/temporal-change stretch goal (Task 23) ✓. Semantic events instead of fixed chunks (Task 18) ✓. Provenance to timestamp/bbox (Task 2 schema, Task 7 OCR bboxes) ✓. Cross-file entity linking (Task 21, works automatically since resolution is source-agnostic) ✓.
- **Placeholder scan:** no TBD/TODO left in any task; every code step has real, runnable code.
- **Type consistency:** `EvidenceUnit`/`Entity`/`KnowledgeEvent`/`Relationship` field names introduced in Task 2 are used identically (including the `metadata_json` rename) through every later task; `RelationshipType`/`EventType`/`EVENT_TYPES` enums defined once (Task 2, Task 22) and referenced, never redefined; `apiClient`/`Citation`/`QueryResponse`/`CompareResponse` from Task 32 are the types every frontend component imports.
- **Scope check:** single cohesive pipeline (ingestion feeds linking feeds retrieval feeds frontend) — not decomposed into independent sub-project specs, since none of the phases ship usefully on their own.

