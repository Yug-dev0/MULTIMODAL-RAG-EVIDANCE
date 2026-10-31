import asyncio
import os
from openai import OpenAI
from sqlalchemy import select
from app.concurrency import call_with_backoff
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
    embeddings = await embed_texts(chunks)  # one batched call instead of one per chunk
    source_type = relevant[0].source_type if relevant else "unknown"
    return [
        await persist_baseline_chunk(session, content_text=c, source_id=source_id,
                                      source_type=source_type, chunk_index=i, embedding=emb)
        for i, (c, emb) in enumerate(zip(chunks, embeddings))
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
    query_embedding = (await embed_texts([query_text]))[0]
    hits = await baseline_vector_search(session, query_embedding, top_k=top_k)
    chunks = [c for c, _ in hits]
    completion = await call_with_backoff(_call_baseline_answer_model, query_text, chunks)
    return completion.choices[0].message.content
