import asyncio
import os
from openai import OpenAI
from app.concurrency import call_with_backoff

MODEL = "text-embedding-3-small"


def _call_embeddings(batch: list[str]):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.embeddings.create(model=MODEL, input=batch)


async def _embed_batch(batch: list[str]) -> list[list[float]]:
    response = await call_with_backoff(_call_embeddings, batch)
    return [item.embedding for item in response.data]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    batches = [texts[i:i + 100] for i in range(0, len(texts), 100)]
    if not batches:
        return []
    results = await asyncio.gather(*[_embed_batch(b) for b in batches])
    return [vec for batch_vecs in results for vec in batch_vecs]
