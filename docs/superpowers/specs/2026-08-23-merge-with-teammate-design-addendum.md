# Addendum: reconciling with teammate's parallel design

Date: 2026-08-23

A teammate independently pushed a design spec + implementation plan to
`origin/main` (`docs/superpowers/specs/2026-08-22-multimodal-rag-pipeline-design.md`,
`docs/superpowers/plans/2026-08-22-multimodal-rag-pipeline.md`) proposing
Postgres+pgvector, Neo4j, Docker Compose, and OpenAI (Whisper API,
GPT-4o-mini, GPT-4o, text-embedding-3-small).

Decision (confirmed with the user): **keep this repo's local-first stack**
(SQLite + Chroma + Kuzu + Ollama/local models) rather than switch to the
teammate's infra — it's already built and smoke-tested end to end (schema,
region-aware OCR, embeddings, all three storage layers), whereas Postgres/
Neo4j/Docker/OpenAI would need to be de-risked from zero with no confirmed
working OpenAI key and no Docker installed. Team is actively coding in
parallel; further remote commits are expected, and will be reconciled
manually rather than via automatic rebase.

Three ideas from the teammate's design are strong enough to adopt into this
stack as enhancements:

1. **`KnowledgeEvent` as a first-class, persisted object.** Previously
   evidence clustering only happened transiently at retrieval time. Instead,
   a new enrichment stage (`src/enrichment/event_assembly.py`) runs after
   cross-modal linking and materializes clusters of tightly-linked
   `EvidenceUnit`s into `KnowledgeEvent` records (title + summary generated
   by one LLM call over the cluster), stored in SQLite, embedded as a third
   Chroma collection (`event_embeddings`), and represented as graph nodes
   with `CONTAINS` edges to their member evidence units. This gives
   retrieval a better default unit (an event already bundles cross-modal
   evidence) and makes the demo narrative far clearer.
2. **Bounded candidate scoring before expensive work.** Cross-modal linking
   must not brute-force all-pairs similarity as the corpus grows. Semantic
   (CLIP) candidate matches are found via Chroma's own ANN search (top-K
   nearest), not a full pairwise sweep; only those top candidates get a
   confidence score and, optionally, an LLM-assigned link label.
3. **A constrained semantic label on cross-modal links**, assigned by one
   bounded LLM call per top candidate pair: `explains | depicts | supports |
   contradicts | follows | references`, stored alongside the existing
   numeric confidence and link_type (`temporal_cooccurrence` /
   `semantic_similarity` / `same_entity`) so retrieval can reason about
   *why* two pieces of evidence are connected, not just that they are.

Not adopted: OpenAI-anything, Postgres/pgvector, Neo4j, Docker, and the
strict `relationship_type` enum for entity-relationships (kept open-
vocabulary per this project's original "don't hardcode" requirement — see
the 2026-08-23 design spec). Contradiction/supersession tracking is noted
as a stretch item, time-permitting, since the demo dataset already has a
natural instance of it (design_notes.pdf's open question, resolved by
addendum_voice_note.wav).
