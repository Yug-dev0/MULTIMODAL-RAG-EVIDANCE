"""
Naive text-only RAG baseline for the required comparison: flattens every
EvidenceUnit's text content into fixed-size CHARACTER chunks (deliberately
naive — the one place in the codebase fixed-size chunking is used), embeds
them with the same text embedding model as the real system for a fair
comparison, and answers via plain top-K similarity search. No graph, no
cross-modal linking, no image channel, no timestamps beyond chunk order.
"""

from __future__ import annotations

import config
from src.enrichment.embeddings import cosine_similarity, embed_text, embed_texts
from src.enrichment.llm_provider import get_llm_provider
from src.storage.sqlite_store import SQLiteStore

TEXTUAL_MODALITIES = {"transcript_segment", "ocr_text", "pdf_text", "video_frame", "image", "pdf_image"}


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    chunk_size = chunk_size or config.BASELINE_CHUNK_SIZE_CHARS
    overlap = config.BASELINE_CHUNK_OVERLAP_CHARS if overlap is None else overlap
    if not text:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size]
        if not piece:
            break
        chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def build_corpus_text(sqlite: SQLiteStore) -> str:
    units = [u for u in sqlite.all() if u.modality.value in TEXTUAL_MODALITIES and u.content.strip()]
    units.sort(key=lambda u: (u.source_file, u.page_number or 0, u.start_time or 0.0))
    return " ".join(u.content for u in units)


def run_baseline_query(query: str, *, sqlite: SQLiteStore, top_k: int | None = None) -> dict:
    top_k = top_k or config.BASELINE_TOP_K
    chunks = chunk_text(build_corpus_text(sqlite))
    if not chunks:
        return {"answer": "No content ingested yet.", "chunks_used": []}

    chunk_vectors = embed_texts(chunks)
    query_vector = embed_text(query)
    ranked = sorted(
        zip(chunks, chunk_vectors), key=lambda pair: cosine_similarity(query_vector, pair[1]), reverse=True
    )
    top_chunks = [c for c, _ in ranked[:top_k]]

    context = "\n---\n".join(top_chunks)
    answer = get_llm_provider().answer_from_context(query, context)
    return {"answer": answer, "chunks_used": top_chunks}
