import os
import uuid as uuid_mod
from fastapi import APIRouter, Depends, HTTPException
from app.api.auth import get_current_user, User
from app.storage.db import get_session
from app.storage.graph import get_driver
from app.storage.repository import (
    corpus_stats, list_sources, delete_source, top_entities, list_knowledge_events, list_evidence_units,
    event_evidence_counts, event_relationship_counts, get_knowledge_event, get_evidence_units_by_event,
    get_entities_by_ids, get_relationships_for_evidence_ids, get_source_session_map,
    list_answered_queries, get_evidence_units_by_ids,
)
from app.ingestion.orchestrator import JOBS
from app.api.history import recent_queries, session_summary
from app.extraction.embeddings import embed_texts
from app.retrieval.vector_search import vector_search_evidence

router = APIRouter()


def _parse_uuids(raw_ids) -> list:
    """Citation ids that are real UUIDs, silently dropping any that aren't.

    citations_json is schemaless JSON written by whatever produced the answer,
    so a single malformed id must not take down the whole page -- the other
    questions and their evidence are still perfectly good.
    """
    parsed = []
    for raw in raw_ids:
        try:
            parsed.append(uuid_mod.UUID(str(raw)))
        except (ValueError, AttributeError, TypeError):
            continue
    return parsed


def _evidence_dict(u, session_map: dict, relevance: float | None) -> dict:
    """One evidence row as the Evidence page consumes it.

    Shared so /evidence and /evidence/by-query can't drift apart -- the page
    renders both through the same card and preview components.
    """
    return {
        "id": str(u.id), "content_text": u.content_text, "modality": u.modality,
        "source_type": u.source_type, "raw_locator": u.raw_locator,
        "timestamp_start": u.timestamp_start, "timestamp_end": u.timestamp_end,
        "page_number": u.page_number, "confidence": u.confidence,
        # Kept distinct from `confidence`: that is how sure extraction was
        # that it read the text correctly, which says nothing about whether
        # the row answers the question.
        "relevance": relevance,
        "extracted_at": u.extracted_at.isoformat() if u.extracted_at else None,
        "session_id": session_map.get(u.source_id, {}).get("id"),
        "session_title": session_map.get(u.source_id, {}).get("title"),
    }


@router.get("/stats/summary")
async def stats_summary():
    async with get_session() as session:
        stats = await corpus_stats(session)
    running = sum(1 for j in JOBS.values() if j.status in ("pending", "running"))
    done = sum(1 for j in JOBS.values() if j.status == "done")
    failed = sum(1 for j in JOBS.values() if j.status == "failed")
    total_jobs = len(JOBS)
    processing_complete_pct = round((done / total_jobs) * 100) if total_jobs else 0
    return {
        **stats,
        "jobs_running": running,
        "jobs_done": done,
        "jobs_failed": failed,
        "jobs_total": total_jobs,
        "processing_complete_pct": processing_complete_pct,
    }


@router.get("/activity/recent")
async def activity_recent(limit: int = 10):
    async with get_session() as session:
        jobs = sorted(JOBS.values(), key=lambda j: j.submitted_at, reverse=True)[:limit]
        activity = []
        for job in jobs:
            modalities: list[str] = []
            evidence_count = 0
            event_count = 0
            if job.source_id:
                from app.storage.repository import get_evidence_units_by_source
                units = await get_evidence_units_by_source(session, uuid_mod.UUID(job.source_id))
                modalities = sorted({u.modality for u in units})
                evidence_count = len(units)
                event_count = len({u.event_id for u in units if u.event_id})
            activity.append({
                "job_id": job.job_id,
                "filename": job.filename,
                "source_type": job.source_type,
                "status": job.status,
                "stage": job.stage,
                "modalities": modalities,
                "evidence_count": evidence_count,
                "event_count": event_count,
                "submitted_at": job.submitted_at,
                "error": job.error,
            })
    return activity


@router.get("/entities/top")
async def entities_top(limit: int = 10):
    async with get_session() as session:
        return await top_entities(session, limit=limit)


@router.get("/sources")
async def sources():
    async with get_session() as session:
        return await list_sources(session)


@router.delete("/sources/{source_id}")
async def delete_source_endpoint(source_id: str):
    try:
        source_uuid = uuid_mod.UUID(source_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="source not found")

    async with get_session() as session:
        result = await delete_source(session, source_uuid)
    if not result["evidence_unit_ids"]:
        raise HTTPException(status_code=404, detail="source not found")

    node_ids = [str(source_uuid)] + [str(i) for i in result["evidence_unit_ids"]] + [str(i) for i in result["event_ids"]]
    driver = get_driver()
    try:
        async with driver.session() as graph_session:
            await graph_session.run(
                "MATCH (n) WHERE n.source_id = $source_id OR n.id IN $node_ids DETACH DELETE n",
                source_id=str(source_uuid), node_ids=node_ids,
            )
    finally:
        await driver.close()

    raw_locator = result["raw_locator"]
    if raw_locator:
        file_path = raw_locator.split("#")[0]
        if os.path.isfile(file_path):
            os.remove(file_path)

    return {"status": "ok"}


@router.get("/events")
async def events(limit: int = 100):
    async with get_session() as session:
        rows = await list_knowledge_events(session, limit=limit)
        evidence_counts = await event_evidence_counts(session)
        relationship_counts = await event_relationship_counts(session, [e.id for e in rows])
    return [
        {
            "id": str(e.id), "event_type": e.event_type.value if hasattr(e.event_type, "value") else e.event_type,
            "title": e.title, "summary": e.summary,
            "timestamp_start": e.timestamp_start, "timestamp_end": e.timestamp_end,
            "confidence": e.confidence, "created_at": e.created_at.isoformat() if e.created_at else None,
            "evidence_count": evidence_counts.get(e.id, 0),
            "relationship_count": relationship_counts.get(e.id, 0),
            "entity_count": len(e.entities or []),
        }
        for e in rows
    ]


@router.get("/events/{event_id}")
async def event_detail(event_id: str):
    async with get_session() as session:
        try:
            event_uuid = uuid_mod.UUID(event_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="event not found")
        event = await get_knowledge_event(session, event_uuid)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        evidence_units = await get_evidence_units_by_event(session, event_uuid)
        evidence_ids = [u.id for u in evidence_units]
        relationships = await get_relationships_for_evidence_ids(session, evidence_ids)
        entities = await get_entities_by_ids(session, list(event.entities or []))

    evidence_by_id = {str(u.id): u for u in evidence_units}
    return {
        "id": str(event.id),
        "event_type": event.event_type.value if hasattr(event.event_type, "value") else event.event_type,
        "title": event.title, "summary": event.summary,
        "timestamp_start": event.timestamp_start, "timestamp_end": event.timestamp_end,
        "confidence": event.confidence,
        "created_at": event.created_at.isoformat() if event.created_at else None,
        "evidence": [
            {
                "id": str(u.id), "content_text": u.content_text, "modality": u.modality,
                "content_type": u.content_type, "raw_locator": u.raw_locator,
                "timestamp_start": u.timestamp_start, "timestamp_end": u.timestamp_end,
                "page_number": u.page_number, "confidence": u.confidence,
            }
            for u in evidence_units
        ],
        "relationships": [
            {
                "id": str(r.id), "source_id": str(r.source_id), "target_id": str(r.target_id),
                "source_label": evidence_by_id.get(str(r.source_id)).content_text[:40] if str(r.source_id) in evidence_by_id else str(r.source_id)[:8],
                "target_label": evidence_by_id.get(str(r.target_id)).content_text[:40] if str(r.target_id) in evidence_by_id else str(r.target_id)[:8],
                "relationship_type": r.relationship_type.value if hasattr(r.relationship_type, "value") else r.relationship_type,
                "confidence": r.confidence,
            }
            for r in relationships
        ],
        "entities": [
            {"id": str(e.id), "canonical_name": e.canonical_name, "entity_type": e.entity_type}
            for e in entities
        ],
    }


@router.get("/evidence")
async def evidence(limit: int = 200, q: str | None = None):
    """Without `q`, lists the corpus newest-first so it can be browsed.

    With `q`, ranks by semantic similarity to the question. The page groups
    rows under their chat session's title, and that title is generated from
    the first question asked in the session -- so an unranked listing reads
    as "here is what answered that question" while actually being "here is
    everything ingested in that session". Ranking is what makes the heading
    tell the truth.
    """
    relevance: dict[str, float] = {}
    async with get_session() as session:
        if q and q.strip():
            query_embedding = (await embed_texts([q]))[0]
            hits = await vector_search_evidence(session, query_embedding, top_k=limit)
            rows = [u for u, _ in hits]
            relevance = {str(u.id): score for u, score in hits}
        else:
            rows = await list_evidence_units(session, limit=limit)
        session_map = await get_source_session_map(session, [u.source_id for u in rows])
    return [_evidence_dict(u, session_map, relevance.get(str(u.id))) for u in rows]


@router.get("/evidence/by-query")
async def evidence_by_query(limit: int = 50, user: User = Depends(get_current_user)):
    """The questions this user asked, each carrying the evidence that answered it.

    Grouping the corpus by chat session put one question's evidence in the same
    bucket as everything else ever ingested in that session, so the answer to
    what was actually asked sat scattered among unrelated rows. The citations
    recorded against each answer are the honest grouping, and they're already
    stored -- nothing surfaced them.
    """
    async with get_session() as session:
        groups = await list_answered_queries(session, owner_uid=user.uid, limit=limit)
        # One lookup for every cited id across all groups, rather than a query
        # per question -- a long chat history would otherwise fan out badly.
        all_ids = {c.get("evidence_id") for g in groups for c in g["citations"] if c.get("evidence_id")}
        units = await get_evidence_units_by_ids(session, _parse_uuids(all_ids))
        session_map = await get_source_session_map(session, [u.source_id for u in units.values()])

    result = []
    for group in groups:
        seen: set[str] = set()
        evidence = []
        for citation in group["citations"]:
            evidence_id = citation.get("evidence_id")
            # Citation order is retrieval rank, so it's preserved. A cited unit
            # can be missing if the evidence was re-ingested since the answer;
            # the question still belongs on the page without it.
            unit = units.get(evidence_id) if evidence_id else None
            if unit is None or evidence_id in seen:
                continue
            seen.add(evidence_id)
            # No relevance score: citations record which evidence was used, not
            # how strongly it matched, and a fabricated number here would be
            # exactly the confusion this page was fixed for.
            evidence.append(_evidence_dict(unit, session_map, None))
        result.append({
            "query": group["query"], "asked_at": group["asked_at"],
            "session_id": group["session_id"], "session_title": group["session_title"],
            "evidence": evidence,
        })
    return result


@router.get("/queries/recent")
async def queries_recent(limit: int = 10):
    return [
        {
            "query": e.query, "mode": e.mode, "citation_count": e.citation_count,
            "modality_count": e.modality_count, "timestamp": e.timestamp,
        }
        for e in recent_queries(limit=limit)
    ]


@router.get("/eval/session-summary")
async def eval_session_summary():
    return session_summary()
