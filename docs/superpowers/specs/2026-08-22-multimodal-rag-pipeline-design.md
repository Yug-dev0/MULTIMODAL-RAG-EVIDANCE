# Multimodal Data Management Pipeline for RAG-Ready Systems — Design

Date: 2026-08-22
Status: Approved

## Problem

Build a pipeline that ingests real multimodal sources (video, audio, images,
PDFs/documents) and produces a structured, retrieval-ready knowledge
representation that preserves cross-modal and temporal relationships between
pieces of evidence — not a text-only RAG pipeline that happens to accept
multimedia inputs. The system must answer queries whose evidence spans
multiple modalities (e.g. "what architecture was discussed for reducing
database load, who explained it, and where was the diagram shown") by
retrieving and connecting evidence across speech, video frames, OCR,
timestamps, and documents — with full provenance back to the original
source. A naive text-only RAG baseline must be built alongside it for
comparison, per the deliverables.

No hardcoding to a specific demo file: ingestion must work generically on
whatever real video/audio/image/PDF a user provides.

## Core data model

Four first-class objects, not a single generic "chunk":

### EvidenceUnit — the raw observation

```
EvidenceUnit {
  id: UUID
  content_text, content_type [transcript|ocr|caption|document_text|image_region]
  modality [speech|video|image|document|ocr]
  source_id, source_type [video|audio|image|pdf|document], raw_locator
  extraction_method, extracted_at
  timestamp_start, timestamp_end
  page_number, bbox [x, y, width, height]
  embedding (vector), entities [entity_id...], keywords
  confidence, confidence_factors (JSONB)
  event_id (nullable, FK -> KnowledgeEvent)
  metadata (JSONB), searchable_text, language
}
```

### KnowledgeEvent — reconstructed meaning spanning evidence

```
KnowledgeEvent {
  id, event_type [discussion|decision|proposal|explanation|
                   diagram_explanation|definition|question|answer|
                   comparison|action_item|observation|reference]
  title, summary
  timestamp_start, timestamp_end
  confidence, entities, created_at, metadata
}
```

Can be multimodal (speech + frame + PDF) or single-modality (speech-only
facts like "meeting starts at 10 AM tomorrow" are still valid events).

### Entity — canonicalized, cross-file

```
Entity { id, canonical_name, entity_type, description, aliases, embedding, metadata }
```

### Relationship — first-class typed edge, not an array field

```
Relationship {
  id
  source_type, source_id, target_type, target_id
  relationship_type [explains|supports|corresponds_to|mentions|depicts|
                      contradicts|follows|references|unrelated]   # strict enum
  confidence
  timestamp_start, timestamp_end
  evidence_ids
  metadata: {
    temporal_overlap, entity_overlap, semantic_similarity,
    vision_model, ...                # auditable component scores, not just a bare confidence
  }
}
```

Postgres holds all four as tables (pgvector on `embedding` columns). Neo4j
mirrors the same objects as a property graph: nodes = EvidenceUnit / Entity /
KnowledgeEvent, edges = materialized Relationship rows.

## Ingestion per modality

- **Video**: ffmpeg extracts audio track; PySceneDetect produces scene-change
  keyframes (not fixed-interval sampling).
- **Audio**: OpenAI Whisper API for transcription with segment timestamps;
  pyannote.audio (local, open-source) for speaker diarization, run as a
  separate pass and merged with Whisper segments by time overlap.
- **Images/frames**: Tesseract OCR (free per-word confidence + bounding
  boxes) + GPT-4o-mini vision for semantic caption/description.
- **PDFs/documents**: PyMuPDF extracts per-page text and embedded images;
  each page's images go through the same OCR+caption path as video frames.

Every extracted item is normalized into an `EvidenceUnit` before anything
else touches it.

## Cross-modal & temporal linking

### Step A — Semantic segmentation (multi-signal, not fixed windows)

A topic boundary in the audio/video timeline is scored from four signals,
combined into a single boundary strength:

```
boundary_strength = f(semantic_similarity_drop, speaker_change,
                       scene_change, pause_duration)
```

A boundary is cut where `boundary_strength` exceeds a threshold, producing
variable-length "semantic windows" instead of fixed 30s chunks.

### Step B — Candidate scoring before any LLM call (cost control)

For each semantic window, candidate keyframes are first filtered to those
whose timestamp falls inside the window, then cheaply scored by:

```
candidate_score = timestamp_proximity + OCR/entity_overlap + embedding_similarity
```

Only the top 1–3 scored candidates per window go to the LLM relationship
call — not an all-pairs sweep. This keeps LLM call volume bounded
regardless of video length.

### Step C — Relationship inference (LLM, strict schema)

One GPT-4o-mini structured-output call per (window, candidate) pair returns:

```json
{
  "related": true,
  "relationship_type": "explains",
  "confidence": 0.93,
  "evidence": "The speaker describes the Redis architecture visible in the frame"
}
```

`relationship_type` is constrained to the enum above — the model cannot
invent new relationship names. The resulting `Relationship` row stores the
component scores from Step B (`temporal_overlap`, `entity_overlap`,
`semantic_similarity`) plus the LLM's own confidence in `metadata`, so every
edge is auditable: "this connection has temporal, semantic, entity, and
model evidence," not a black-box similarity number.

Same pattern (PDF text vs. an evidence unit) produces `supports` /
`corresponds_to` relationships between documents and speech/frames.

### Step D — Entity resolution (safe fallback, not pure LLM)

```
raw mention -> normalize -> alias lookup
  found -> use existing Entity
  not found -> embedding search
    similarity > threshold -> use existing Entity
    otherwise               -> create new Entity
```

Borderline-similarity mentions do not spawn duplicate entities by default;
the threshold is tuned conservative to bias toward merging.

### Step E — KnowledgeEvent assembly (deterministic scoring, not vibes)

```
EventScore = 0.35*temporal_overlap + 0.30*entity_overlap
           + 0.20*semantic_similarity + 0.15*relationship_confidence
```

EvidenceUnits whose pairwise `EventScore` clears a threshold are merged into
the same `KnowledgeEvent`. Single-modality clusters are valid events.
`event_type`, `title`, `summary` are generated by one LLM call over the
cluster's evidence, `event_type` constrained to the enum in the schema.

### Step F — Contradiction / supersession tracking

When two EvidenceUnits/Events about the same Entity conflict (e.g. "we'll
use Redis" then later "we decided not to use Redis"), a `contradicts`
Relationship is created between them. Retrieval-time resolution (below)
uses this to prefer the temporally later claim while still surfacing the
superseded one, flagged — enabling "what was the final decision" queries.

## Retrieval

1. Query embedded (text-embedding-3-small) and NER'd.
2. Three parallel candidate paths:
   a. Vector search over `EvidenceUnit.embedding`.
   b. Vector search over `KnowledgeEvent` summary embeddings (events already
      bundle cross-modal evidence, often the better retrieval unit).
   c. Entity path — query entities resolved through the Step D pipeline,
      then graph-pulled via Neo4j for everything linked to them.
3. Graph expansion: 1–2 hop Neo4j traversal from top candidates over
   Relationship edges.
4. Contradiction resolution: prefer most-recent claim on conflicting
   entities, surface superseded ones flagged.
5. Fusion + rerank: weighted(vector similarity, graph/relationship
   confidence, entity match, recency-for-contradictions).
6. Answer synthesis: GPT-4o given top KnowledgeEvents + their evidence
   (frame images as vision input, OCR, doc snippets, transcript text) ->
   grounded natural-language answer with inline citations to specific
   EvidenceUnit ids.
7. Response payload:
   `{answer, citations: [{evidence_id, modality, source, timestamp/page,
   thumbnail, confidence}], events_used}`.

Frontend renders the answer plus evidence cards (video seeks to timestamp,
PDF opens to page/bbox) — full traceability to source, per requirement D.

## Baseline text-only RAG (for the required comparison)

A deliberately naive sibling pipeline, isolated in `backend/app/baseline/`:
fixed-size chunking (500 tokens / 50 overlap) of all extracted text
(transcript + OCR + doc text) with no entity/event/graph structure, stored
in a separate `baseline_chunks` pgvector table, plain top-K vector search ->
LLM answer. The same eval query set (`eval/queries.json`) runs against both
systems; `eval/report.md` documents where the baseline fails to retrieve or
connect cross-modal evidence that the multimodal system gets right.

## Tech stack

| Concern | Choice |
|---|---|
| Backend | Python 3.11, FastAPI, in-process background jobs (no Celery/Redis) |
| ASR | OpenAI Whisper API |
| Diarization | pyannote.audio (local) |
| OCR | Tesseract |
| Vision/caption | GPT-4o-mini |
| PDF | PyMuPDF |
| Video | ffmpeg + PySceneDetect |
| Embeddings | OpenAI text-embedding-3-small |
| Structured extraction (NER/relationship/event) | GPT-4o-mini, strict JSON-schema outputs |
| Answer synthesis | GPT-4o (multimodal) |
| Relational + vector store | Postgres 16 + pgvector |
| Graph store | Neo4j 5 Community |
| Frontend | React + Vite + TS, `3d-force-graph`, Tailwind, PDF.js, native `<video>` |
| Infra | single `docker-compose.yml` (postgres, neo4j, backend, frontend) |

## Repo structure

```
/backend/app/ingestion       video.py, audio.py, image.py, pdf.py, orchestrator.py
/backend/app/extraction      asr.py, diarization.py, ocr.py, vision.py, ner.py
/backend/app/linking         segmentation.py, candidate_scoring.py,
                              relationship_inference.py, entity_resolution.py,
                              event_assembly.py, contradiction.py
/backend/app/storage         postgres models (SQLAlchemy), neo4j client, pgvector helpers
/backend/app/retrieval       vector_search.py, graph_expansion.py, fusion_rerank.py,
                              answer_synthesis.py
/backend/app/baseline        naive_chunk_rag.py
/backend/app/api             routers: ingest, query, compare, graph, jobs
/backend/app/schemas         pydantic models mirroring the four core objects
/frontend/src/components     QueryPanel, EvidenceCard, GraphView(3D), VideoPlayer,
                              PDFViewer, CompareView
/eval                        queries.json, run_eval.py, report.md
docker-compose.yml, README.md, .env.example
```

## Testing / eval plan

- Unit tests on pure scoring functions (`EventScore`, candidate scoring,
  boundary strength) with mocked LLM responses — no API calls needed for
  these.
- Integration test: a small fixture video/PDF run through the full pipeline
  locally, asserting EvidenceUnits/Entities/Events get created with
  expected shapes.
- Eval set: ~8–10 hand-labeled queries against the real demo dataset, each
  tagged with which modalities the correct answer requires, run through
  both the multimodal system and the baseline, written up in
  `eval/report.md`.

## Explicitly out of scope (per problem statement)

Training custom foundation models; building ASR/OCR/vision models from
scratch; production-grade distributed infra; a hand-built vector database;
full auth; support for every media format; a fully polished frontend beyond
what's needed for a strong live demo.

## Open risks / fallbacks

- pyannote.audio's diarization model is gated on HuggingFace and needs a
  free token + one-time terms acceptance. Fallback if this blocks: drop
  speaker-change as a segmentation signal and rely on semantic similarity +
  scene change + pause duration only.
- LLM call volume is bounded by the candidate-scoring funnel (Step B), but
  very long videos should still be capped (e.g. max keyframes considered)
  to keep ingestion time and OpenAI cost predictable during the demo.
