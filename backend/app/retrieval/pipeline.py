import asyncio
from sqlalchemy import select
from app.extraction.embeddings import embed_texts
from app.retrieval.vector_search import vector_search_evidence, vector_search_events
from app.retrieval.graph_expansion import entities_for_query, evidence_linked_to_entities, graph_expand
from app.retrieval.fusion_rerank import fuse_and_rerank
from app.retrieval.answer_synthesis import synthesize_answer
from app.storage.db import get_session
from app.storage.models import EvidenceUnit


async def _vector_search_evidence_isolated(query_embedding, top_k, allowed_source_ids):
    async with get_session() as s:
        return await vector_search_evidence(s, query_embedding, top_k=top_k, allowed_source_ids=allowed_source_ids)


async def _vector_search_events_isolated(query_embedding, top_k, allowed_source_ids):
    async with get_session() as s:
        return await vector_search_events(s, query_embedding, top_k=top_k, allowed_source_ids=allowed_source_ids)


async def _entities_for_query_isolated(query_text):
    async with get_session() as s:
        return await entities_for_query(s, query_text)


async def run_query(session, driver, query_text: str, top_k: int = 10, allowed_source_ids: list | None = None):
    query_embedding = (await embed_texts([query_text]))[0]

    # Previously these three ran one after another on the shared route
    # session. A single AsyncSession can't be used from concurrent
    # coroutines, so each gets its own isolated session here instead — that
    # makes it safe to run them concurrently, overlapping their latency
    # rather than paying for it three times in a row.
    evidence_hits, event_hits, entities = await asyncio.gather(
        _vector_search_evidence_isolated(query_embedding, top_k, allowed_source_ids),
        _vector_search_events_isolated(query_embedding, top_k, allowed_source_ids),
        _entities_for_query_isolated(query_text),
    )
    vector_hits = {str(u.id): sim for u, sim in evidence_hits}
    seed_ids = list(vector_hits.keys())

    # Both are independent Neo4j reads (each opens its own driver session
    # internally) — safe and cheap to run together even when one side's
    # input list is empty (an empty `IN []` just matches nothing).
    entity_hit_ids_raw, graph_hit_ids_raw = await asyncio.gather(
        evidence_linked_to_entities(driver, [e.id for e in entities]),
        graph_expand(driver, seed_ids, hops=2),
    )
    entity_hit_ids = set(entity_hit_ids_raw)
    graph_hit_ids = set(graph_hit_ids_raw)

    confidences = {str(u.id): u.confidence for u, _ in evidence_hits}
    ranked = fuse_and_rerank(
        vector_hits=vector_hits, graph_hit_ids=graph_hit_ids,
        entity_hit_ids=entity_hit_ids, confidences=confidences,
    )

    top_ids = [r.id for r in ranked[:top_k]]
    if top_ids:
        # Graph/entity expansion can pull in ids from outside the vector
        # hits (and thus outside allowed_source_ids) -- re-apply the scope
        # here so a scoped chat session can never surface another topic's
        # evidence, regardless of which signal originally proposed it.
        stmt = select(EvidenceUnit).where(EvidenceUnit.id.in_(top_ids))
        if allowed_source_ids is not None:
            stmt = stmt.where(EvidenceUnit.source_id.in_(allowed_source_ids))
        result = await session.execute(stmt)
        by_id = {str(u.id): u for u in result.scalars().all()}
        ranked_evidence = [by_id[i] for i in top_ids if i in by_id]
    else:
        ranked_evidence = [u for u, _ in evidence_hits]

    ranked_events = [e for e, _ in event_hits]
    return await synthesize_answer(query_text, ranked_evidence, ranked_events)
