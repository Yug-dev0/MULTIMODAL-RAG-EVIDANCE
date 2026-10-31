# Multimodal Data Management Pipeline for RAG-Ready Systems — Design Spec

Date: 2026-08-23
Status: Approved for implementation planning

## 1. Problem summary

Hackathon brief (13-hour build): build a data-management pipeline that turns raw
multimodal sources (video, audio, images, PDFs) into structured, retrieval-ready
knowledge that **preserves relationships between modalities and time**, not just
converted-to-text chunks in a vector store. The bar judges apply: a system that
can answer "what architecture was discussed for reducing database load, who
explained it, and where was the diagram shown" by actually fusing speech +
visual + document evidence — not by doing ordinary text-only vector search.

Evaluation weights: multimodal ingestion quality (20%), data representation
quality (20%), cross-modal/temporal preservation (20%), retrieval quality (20%),
technical depth/innovation (15%), demo quality (5%).

Explicitly out of scope (per brief): training foundation models, building
ASR/OCR/vision models from scratch, production distributed infra, building a
vector DB from scratch, auth, universal format support, polished frontend.

## 2. Guiding principle: no hardcoding

The pipeline must be fully generic and config/data-driven. Adding a new video,
audio file, image, or PDF means adding a row to `configs/dataset.yaml` — never
touching extraction/linking code. No source-specific branches, no hand-entered
timestamps, entity lists, or synonym tables anywhere in `src/`. Entity
resolution, cross-modal linking, and relationship extraction are all derived
programmatically from the actual content of each run.

The one legitimate exception: `src/mmrag/eval/queries.yaml`, a small
hand-labeled set of test queries + gold relevant-evidence IDs used to measure
retrieval quality. This is a standard evaluation fixture (like a test corpus),
not pipeline logic, and must not be confused with hardcoding — this distinction
is called out explicitly in the README.

## 3. Dataset

Real, verified, openly-licensed sources — thematically chosen so the **same
architecture is discussed in both spoken/visual form and written form**,
mirroring the brief's own example query:

| # | Video (real lecture, real whiteboard/slide diagrams) | Paper (same architecture, written) |
|---|---|---|
| 1 | [MIT 6.824 Lecture 3 — GFS](https://www.youtube.com/watch?v=EpIgvowZr00) | [The Google File System, SOSP'03](http://research.google.com/archive/gfs-sosp2003.pdf) |
| 2 | [MIT 6.824 Lecture 6 — Raft](https://www.youtube.com/watch?v=64Zp3tzNbpE) | [In Search of an Understandable Consensus Algorithm](https://raft.github.io/raft.pdf) |
| 3 | [MIT 6.824 Lecture 16 — Memcached at Facebook](https://www.youtube.com/watch?v=Myp8z0ybdzM) | [Scaling Memcache at Facebook, NSDI'13](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final170_update.pdf) |

Course site: https://pdos.csail.mit.edu/6.824/ (CC BY 3.0 US). Papers are
open-access PDFs hosted by their original publishers/authors (USENIX open
access, Google Research archive, official Raft site). All links verified live
via web search/fetch on 2026-08-23, not recalled from memory.

**Flagship demo query:** "What architecture was discussed for reducing
database load, who explained it, and where was the corresponding diagram
shown?" — this is answerable only from lecture 16 + its paper, and the
course's own lecture notes describe the memcache layer as producing a "100x
reduction in DB read load," i.e. this is near word-for-word the brief's own
example.

Rollout: build and validate the entire pipeline end-to-end against source #1
(GFS) first. Only after it's proven correct do we ingest #2 and #3. All three
ship in the final demo.

Video download capped at 720p (`yt-dlp -f "best[height<=720]"`) to bound disk
use; audio extracted via `ffmpeg`. Given ~10GB free disk on the build machine,
model footprint is kept lean (see §9 Risks).

## 4. Architecture approaches considered

**A — Graph-anchored hybrid retrieval (chosen).** Extraction produces typed
Knowledge Units; a knowledge graph stores entities + relationships + temporal
edges between units; a vector index stores embeddings with rich metadata.
Retrieval fuses vector similarity with graph traversal. Strongest on the two
highest-weighted judge criteria because relationships are explicit and
queryable, not implicit in chunk adjacency.

**B — Metadata-filtered vector store only, no graph DB.** Same extraction, but
relationships become ad hoc timestamp-window joins computed at query time.
Simpler, but no persisted/queryable structure for cross-file entity
relationships or a temporal knowledge graph (both stretch goals), and nothing
to visualize in the demo.

**C — Agentic/LLM-orchestrated retrieval.** An LLM with search tools decides
what to look up at query time. Flexible, but provenance is harder to guarantee
and reproduce, it's slower, and it demonstrates less about *how information is
structured* — which is exactly what judges are told to evaluate.

**Decision:** Approach A. B's temporal-window joins are reused as one of
several edge-creation rules inside A. C's idea survives only as an optional
final layer: an LLM writes a cited natural-language summary **from** the
already-retrieved, already-traceable evidence bundle — never as the retrieval
mechanism itself.

## 5. Pipeline stages

1. **Acquire** — `yt-dlp` fetches each lecture video; `ffmpeg` extracts a WAV
   audio track and a keyframe stream. Every source file is content-hashed
   (SHA-256) and registered in `sources`; re-running the pipeline on an
   already-hashed source is a no-op (idempotent, safe to re-run during a
   hackathon).
2. **Per-modality extraction**
   - *Audio* → `faster-whisper` (local, CPU, int8 quantized `small` model,
     re-benchmarked on source #1 before deciding whether `base` or `medium` is
     the better speed/quality tradeoff for the remaining sources) → segments
     with word-level timestamps + `avg_logprob`-derived confidence.
   - *Video* → scene-change/frame-diff keyframe selection (PySceneDetect or
     perceptual-hash frame diff) — keyframes are emitted only on meaningful
     visual change (slide flip, new diagram drawn), not fixed intervals. This
     **is** the "semantic events, not fixed-size chunks" stretch goal applied
     to the visual channel.
   - *Keyframes* → Tesseract OCR for exact on-screen text with bounding boxes
     + native confidence (precise provenance) **and** Gemini vision
     (`gemini-2.5-flash`) for semantic diagram/scene understanding (what the
     diagram *means*). Two different tools for two different jobs,
     deliberately not conflated.
   - *PDFs* → PyMuPDF for text blocks (page number + bbox) **and** embedded
     figure/image extraction, so a paper's architecture diagram becomes its
     own image-modality evidence unit, linkable to the same diagram drawn live
     in the lecture.
3. **Normalize** — every extracted item becomes a Knowledge Unit (schema in
   §6). This uniform schema is what satisfies "Structured Knowledge
   Representation."
4. **Enrich & link**
   - spaCy (`en_core_web_sm`) gives cheap generic NER on every unit as a fast
     deterministic baseline layer.
   - Gemini structured JSON-schema extraction runs per semantic segment
     (a temporally/visually coherent cluster of units, not a raw sentence) to
     pull domain entities and relationships, e.g. `Memcache --mitigates-->
     DatabaseLoad`, each tagged with the evidence unit IDs that support it and
     a self-reported confidence (explicitly documented as heuristic, not a
     calibrated probability — see §9).
   - Entity resolution coalesces aliases ("the master" / "GFS master") via
     embedding similarity + string normalization — never a hardcoded synonym
     table.
   - Cross-modal edges are created by two generic rules, both config-driven
     (window size, hop limits — not magic numbers buried in code):
     1. **Temporal co-occurrence**: a speech unit and a visual unit from the
        same source within a configurable window (default ±20s) get a
        `CO_OCCURS_WITH` edge, confidence scaled by proximity.
     2. **Entity coreference**: any two units (any source, any modality) that
        share a resolved entity get linked via that entity node — this is the
        mechanism that connects a lecture's spoken explanation, its on-screen
        diagram, and the paper's figure, even across files.
5. **Store** (details in §7).
6. **Retrieve** (details in §8).
7. **Evaluate** — see §9 for methodology, metrics, and baseline design.
8. **Serve** — FastAPI (`/ingest`, `/query`, `/graph/{entity}`) + Streamlit UI:
   query box, evidence cards per modality (video snippet that seeks to
   timestamp, frame thumbnail with OCR bbox highlighted, PDF page excerpt with
   highlighted span), an entity-neighborhood graph view, and a
   baseline-vs-multimodal comparison toggle.

## 6. Structured schema

**`sources`** — `id, file_path, original_url, title, source_type
{video|pdf}, license, content_hash, duration_seconds, page_count, ingested_at`.

**`knowledge_units`** (the core structured representation):

| field | type | notes |
|---|---|---|
| `id` | UUID | primary key |
| `content` | text | transcript sentence, OCR text, Gemini caption, or PDF chunk |
| `modality` | enum | `speech`, `visual_frame`, `visual_ocr`, `document_text`, `document_figure` |
| `source_id` | FK | → `sources.id` |
| `start_ts`, `end_ts` | float, nullable | seconds, for video/audio-derived units |
| `page_number` | int, nullable | for PDF-derived units |
| `bbox` | JSON, nullable | `[x0,y0,x1,y1]` — OCR region, PDF text block, or figure crop |
| `asset_path` | str, nullable | path to extracted frame/figure image |
| `entities` | JSON list | resolved entity IDs mentioned in this unit |
| `confidence` | float 0-1 | see §9 for how each modality derives this |
| `extraction_method` | str | e.g. `faster-whisper-small`, `tesseract-5`, `gemini-2.5-flash-vision`, `pymupdf` |
| `embedding_id` | str | pointer into the Qdrant point for this unit |
| `created_at` | timestamp | |

**`entities`** — `id, canonical_name, entity_type` (free string suggested by
the LLM from an open vocabulary, not a hardcoded enum), `aliases (JSON list)`,
`first_seen_unit_id`.

**`relationships`** — `id, subject_entity_id, predicate, object_entity_id
(nullable), evidence_unit_ids (JSON list), relation_source
{temporal_cooccurrence|llm_extracted|entity_coreference}, confidence,
valid_at_ts, created_at`.

## 7. Storage design

| Layer | Tech | Why |
|---|---|---|
| Canonical ledger | **SQLite** via SQLAlchemy | Zero setup, ground truth for every Knowledge Unit / Source / Relationship, trivially exportable/queryable. |
| Vector index | **Qdrant, embedded/local mode** (`QdrantClient(path=...)`, no server/Docker) | Payload-filterable (modality, source, time range, entities, confidence) hybrid search; chosen over Chroma for richer filtering, over server-mode Qdrant because Docker isn't installed on this machine. |
| Knowledge graph | **Kuzu, embedded** (no server/Docker) | Real Cypher queries, temporal edge properties, persists to disk — gives the temporal-knowledge-graph stretch goal and a demoable subgraph view with zero infra risk on a Docker-less, disk-constrained machine. |

Qdrant collection `knowledge_units`: vector field from `gemini-embedding-001`
(explicit `output_dimensionality` set, e.g. 768, to control index size);
payload mirrors the SQLite record's filterable fields.

Kuzu schema: node tables `Entity(id, canonical_name, entity_type)` and
`Unit(id, modality, source_id, start_ts, confidence)`; relationship tables
`MENTIONS(Unit→Entity, confidence)`, `RELATES_TO(Entity→Entity, predicate,
confidence, valid_at_ts, relation_source)`, `CO_OCCURS_WITH(Unit→Unit,
confidence, time_delta)`.

## 8. Retrieval design

1. Embed the query with `gemini-embedding-001`.
2. Vector search in Qdrant across all modalities (optionally payload-filtered,
   e.g. "only visual evidence").
3. Graph expansion in Kuzu: from the entities attached to top vector hits,
   traverse `RELATES_TO`/`MENTIONS`/`CO_OCCURS_WITH` up to a configurable hop
   limit (default 2) to pull in connected evidence that may not have scored
   highly on pure text similarity — this is the concrete mechanism that
   recovers a diagram frame from a transcript-only query.
4. Fuse and rank:
   `final_score = w_v * vector_similarity + w_g * graph_proximity_boost(hops) + w_c * confidence`,
   default weights `w_v=0.6, w_g=0.3, w_c=0.1` exposed in config and tunable
   against the gold query set (§9) — not magic numbers buried in code.
5. Return an evidence bundle: each item carries its full provenance
   (source, modality, timestamp/page/bbox, confidence, extraction method).
6. Optional: Gemini writes a short cited natural-language answer strictly
   from the returned bundle (never expands retrieval itself).

## 9. Evaluation

**Baseline ("naive text RAG"):** flatten every source to plain text
(transcript + OCR text + PDF text), fixed 500-token chunks, single embedding
index (same embedding model, for a fair comparison), plain top-K vector
search, no graph, no timestamps beyond chunk order, no cross-modal linking.

**Gold query set** (`src/mmrag/eval/queries.yaml`, built *after* source #1 is
ingested so real unit IDs exist to label): ~12-15 queries spanning (a)
cross-modal fusion queries requiring speech+visual+document evidence together
— including the flagship "database load" query — (b) plain factual control
queries answerable from text alone, (c) temporal queries ("what was discussed
right before/after X"), (d) cross-file entity queries ("how does the paper's
description of X compare to the lecture's explanation").

**Metrics:** Recall@5 / Recall@10 against hand-labeled relevant unit IDs for
both systems, plus a **modality-coverage** metric — for queries whose gold
set spans ≥2 modalities, does the system's top-K also span ≥2 modalities?
This operationalizes the actual hackathon challenge (cross-modal retrieval),
not just generic recall, and is the headline chart for the demo.

**Confidence, honestly scoped:** ASR confidence comes from Whisper's
`avg_logprob` (a real model-internal signal); OCR confidence is Tesseract's
native per-word confidence (real); Gemini's self-reported extraction
confidence is a heuristic the model is asked to output, not a calibrated
probability — this is stated plainly in the README rather than oversold.

## 10. Repo structure

```
configs/dataset.yaml          # sources, licenses — no logic
src/mmrag/ingest/             # video.py, audio_asr.py, vision_ocr.py, pdf_doc.py, download.py
src/mmrag/schema/             # KnowledgeUnit / Entity / Relationship models (pydantic)
src/mmrag/enrich/             # ner.py, llm_extract.py, linking.py
src/mmrag/store/              # relational.py (SQLite), vectorstore.py (Qdrant), graphstore.py (Kuzu)
src/mmrag/retrieve/           # hybrid.py, baseline.py
src/mmrag/eval/               # queries.yaml, evaluate.py
src/mmrag/pipeline.py         # resumable CLI orchestrator
app/api.py                    # FastAPI
app/ui_streamlit.py           # Streamlit demo UI
docs/architecture.md          # for the deliverable: architecture + key decisions
docs/future_improvements.md   # for the deliverable: future improvements
.env (gitignored), .env.example
```

## 11. Risks and mitigations

- **~10GB free disk.** Mitigation: 720p video cap, keyframes-not-every-frame,
  no local CLIP/vision-transformer weights in the core build (Gemini handles
  vision over the network instead), monitor free space during ingestion and
  pause if it gets tight.
- **~4 hours of audio to transcribe locally, in full.** Mitigation:
  benchmark ASR speed on source #1 before committing to sources #2 and #3;
  run long transcription jobs in the background; pipeline is resumable via
  the source content-hash registry so nothing already processed is redone.
- **Secrets.** The Gemini key goes only into a local, git-ignored `.env`;
  never printed again, never committed; `.env.example` documents the required
  variable name with a placeholder.
- **Gemini quota/latency.** Use `flash`-tier models by default (fast, cheap),
  add retry/backoff, make model name configurable rather than hardcoded so it
  can be swapped if needed.

## 12. Stretch goals covered

Semantic event segmentation (not fixed chunks) ✅ · cross-file/cross-modal
entity relationships ✅ · exact provenance to timestamp/bbox ✅ ·
multi-evidence-type single-query retrieval ✅ · temporal knowledge graph ✅ ·
text-only RAG comparison with measurable metrics ✅ · confidence scores used
during retrieval ranking ✅ · "how information changes over time" — partially,
via entity-mention timelines across the three sources rather than a fully
separate subsystem (explicitly scoped down given the 13-hour frame).

## 13. Future improvements (for the deliverable write-up)

- Speaker diarization for multi-speaker segments (pyannote.audio) — attribution
  currently assumes a single primary lecturer per source.
- A true joint image/text embedding space (CLIP-style) for pixel-level visual
  similarity search independent of OCR/caption text, layered in after the core
  pipeline is proven, disk permitting.
- A learned fusion re-ranker replacing the fixed-weight vector/graph/confidence
  formula, once enough query/feedback data exists to train one.
- Incremental/streaming ingestion for live meetings rather than batch-only.
- Calibrated (not just self-reported) confidence scores.
