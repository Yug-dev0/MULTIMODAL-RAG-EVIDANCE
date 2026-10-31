from sqlalchemy import select, func
from app.storage.models import EvidenceUnit, KnowledgeEvent

# Transcript segmentation emits stubs -- single tokens like "AI." or "Okay." --
# that carry almost no meaning but embed very close to any short question, so
# they outranked substantive passages and filled the top of every result list.
# A unit this short can't support a claim regardless of its score. Deliberately
# low: terse-but-real evidence ("Revenue: $4.2M") must still survive, so this
# only excludes what is unusable by any reading.
MIN_EVIDENCE_CHARS = 10


async def vector_search_evidence(session, query_embedding: list[float], top_k: int = 10,
                                  allowed_source_ids: list | None = None):
    distance = EvidenceUnit.embedding.cosine_distance(query_embedding)
    stmt = select(EvidenceUnit, distance.label("distance")).where(
        EvidenceUnit.embedding.is_not(None),
        func.length(func.trim(EvidenceUnit.content_text)) >= MIN_EVIDENCE_CHARS,
    )
    if allowed_source_ids is not None:
        stmt = stmt.where(EvidenceUnit.source_id.in_(allowed_source_ids))
    stmt = stmt.order_by(distance).limit(top_k)
    result = await session.execute(stmt)
    return [(row.EvidenceUnit, 1.0 - row.distance) for row in result]


async def vector_search_events(session, query_embedding: list[float], top_k: int = 10,
                                allowed_source_ids: list | None = None):
    distance = KnowledgeEvent.embedding.cosine_distance(query_embedding)
    stmt = select(KnowledgeEvent, distance.label("distance")).where(KnowledgeEvent.embedding.is_not(None))
    if allowed_source_ids is not None:
        # An event has no source_id of its own -- scope it via its member
        # evidence units instead, so cross-topic events can't surface either.
        # A subquery (rather than a join) sidesteps DISTINCT on a row that
        # includes a plain `json` column, which Postgres can't compare.
        scoped_event_ids = select(EvidenceUnit.event_id).where(
            EvidenceUnit.source_id.in_(allowed_source_ids), EvidenceUnit.event_id.is_not(None)
        )
        stmt = stmt.where(KnowledgeEvent.id.in_(scoped_event_ids))
    stmt = stmt.order_by(distance).limit(top_k)
    result = await session.execute(stmt)
    return [(row.KnowledgeEvent, 1.0 - row.distance) for row in result]
