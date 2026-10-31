import enum
import uuid
from collections import Counter
from datetime import datetime
from sqlalchemy import delete, func, select, update
from app.extraction.embeddings import embed_texts
from app.storage.models import (
    EvidenceUnit, Entity, KnowledgeEvent, Relationship, BaselineChunk,
    ChatSession, ChatMessage, SessionSource,
)


def _graph_safe(value):
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_graph_safe(v) for v in value]
    return value


def model_to_dict(instance) -> dict:
    return {c.name: _graph_safe(getattr(instance, c.name)) for c in instance.__table__.columns}


async def persist_evidence_unit(session, *, content_text, content_type, modality, source_id,
                                 source_type, raw_locator, extraction_method,
                                 timestamp_start=None, timestamp_end=None, page_number=None,
                                 bbox=None, confidence, confidence_factors=None, event_id=None,
                                 metadata_json=None, language="en", embedding=None) -> EvidenceUnit:
    # Callers persisting many units in a loop (e.g. one per transcript
    # segment) should batch-embed all their texts in a single call up front
    # and pass the result in here — embedding one text at a time turns what
    # could be one API round-trip into N of them. Falls back to embedding
    # just this text when the caller doesn't have a precomputed vector.
    if embedding is None:
        embedding = (await embed_texts([content_text]))[0]
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
    embedding = (await embed_texts([canonical_name]))[0]
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
    embedding = (await embed_texts([summary]))[0]
    event = KnowledgeEvent(
        id=uuid.uuid4(), event_type=event_type, title=title, summary=summary,
        timestamp_start=timestamp_start, timestamp_end=timestamp_end, embedding=embedding,
        confidence=confidence, entities=entities or [], created_at=datetime.utcnow(),
        metadata_json=metadata_json or {},
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


async def persist_baseline_chunk(session, *, content_text, source_id, source_type, chunk_index,
                                  embedding=None) -> BaselineChunk:
    if embedding is None:
        embedding = (await embed_texts([content_text]))[0]
    chunk = BaselineChunk(
        id=uuid.uuid4(), content_text=content_text, source_id=source_id, source_type=source_type,
        chunk_index=chunk_index, embedding=embedding, metadata_json={},
    )
    session.add(chunk)
    await session.flush()
    return chunk


def _utc_iso(dt: datetime) -> str:
    # created_at/updated_at are stored as naive UTC (datetime.utcnow()); a
    # plain .isoformat() has no offset, so browsers parse it as local time
    # and every relative timestamp in the UI drifts by the viewer's UTC
    # offset. Tag it explicitly so it round-trips as UTC on the client.
    return dt.isoformat() + "Z"


async def create_chat_session(session, *, owner_uid: str, title: str = "New chat") -> ChatSession:
    chat_session = ChatSession(id=uuid.uuid4(), owner_uid=owner_uid, title=title)
    session.add(chat_session)
    await session.flush()
    return chat_session


async def get_chat_session(session, session_id, owner_uid: str) -> ChatSession | None:
    result = await session.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.owner_uid == owner_uid)
    )
    return result.scalar_one_or_none()


async def list_chat_sessions(session, owner_uid: str) -> list[dict]:
    sessions = (
        await session.execute(
            select(ChatSession).where(ChatSession.owner_uid == owner_uid).order_by(ChatSession.updated_at.desc())
        )
    ).scalars().all()
    if not sessions:
        return []
    session_ids = [s.id for s in sessions]
    msg_counts = dict(
        (await session.execute(
            select(ChatMessage.session_id, func.count(ChatMessage.id))
            .where(ChatMessage.session_id.in_(session_ids)).group_by(ChatMessage.session_id)
        )).all()
    )
    source_counts = dict(
        (await session.execute(
            select(SessionSource.session_id, func.count(SessionSource.source_id))
            .where(SessionSource.session_id.in_(session_ids)).group_by(SessionSource.session_id)
        )).all()
    )
    return [
        {
            "id": str(s.id), "title": s.title,
            "created_at": _utc_iso(s.created_at), "updated_at": _utc_iso(s.updated_at),
            "message_count": msg_counts.get(s.id, 0), "source_count": source_counts.get(s.id, 0),
        }
        for s in sessions
    ]


async def touch_chat_session(session, session_id, *, title: str | None = None) -> None:
    values = {"updated_at": datetime.utcnow()}
    if title is not None:
        values["title"] = title
    await session.execute(update(ChatSession).where(ChatSession.id == session_id).values(**values))


async def link_source_to_session(session, *, source_id, session_id) -> None:
    session.add(SessionSource(source_id=source_id, session_id=session_id))


async def get_session_source_ids(session, session_id) -> list[uuid.UUID]:
    result = await session.execute(select(SessionSource.source_id).where(SessionSource.session_id == session_id))
    return [row[0] for row in result]


async def persist_chat_message(session, *, session_id, role: str, content: str, citations_json=None) -> ChatMessage:
    message = ChatMessage(
        id=uuid.uuid4(), session_id=session_id, role=role, content=content, citations_json=citations_json or [],
    )
    session.add(message)
    await session.flush()
    return message


async def get_chat_messages(session, session_id) -> list[ChatMessage]:
    result = await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    )
    return result.scalars().all()


async def list_answered_queries(session, owner_uid: str, limit: int = 50) -> list[dict]:
    """Every question this user asked, paired with the citations of the answer
    it got, newest first.

    The pairing is positional: an assistant message answers the user message
    immediately before it in the same session. That is how the chat writes
    them (see api/sessions.session_query), and it's why a question with no
    answer yet -- a request that failed mid-flight -- simply drops out rather
    than being paired with some unrelated later answer.
    """
    result = await session.execute(
        select(ChatMessage, ChatSession.id, ChatSession.title)
        .join(ChatSession, ChatSession.id == ChatMessage.session_id)
        .where(ChatSession.owner_uid == owner_uid)
        .order_by(ChatMessage.session_id, ChatMessage.created_at)
    )
    rows = list(result)

    queries: list[dict] = []
    pending: dict | None = None
    for message, session_id, session_title in rows:
        if message.role == "user":
            pending = {
                "query": message.content, "asked_at": _utc_iso(message.created_at),
                "session_id": str(session_id), "session_title": session_title,
                "citations": [],
            }
        elif message.role == "assistant" and pending is not None:
            pending["citations"] = message.citations_json or []
            queries.append(pending)
            pending = None

    queries.sort(key=lambda q: q["asked_at"], reverse=True)
    return queries[:limit]


async def get_evidence_units_by_ids(session, evidence_ids: list) -> dict:
    """id -> EvidenceUnit for the ids that still exist.

    Returned as a map because callers need to preserve their own ordering
    (citation rank), not whatever order the database hands back.
    """
    if not evidence_ids:
        return {}
    result = await session.execute(select(EvidenceUnit).where(EvidenceUnit.id.in_(evidence_ids)))
    return {str(u.id): u for u in result.scalars().all()}


async def get_evidence_units_by_source(session, source_id) -> list[EvidenceUnit]:
    result = await session.execute(select(EvidenceUnit).where(EvidenceUnit.source_id == source_id))
    return list(result.scalars().all())


async def list_entities(session, limit: int | None = None) -> list[Entity]:
    stmt = select(Entity)
    if limit:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_knowledge_events(session, limit: int = 100) -> list[KnowledgeEvent]:
    result = await session.execute(
        select(KnowledgeEvent).order_by(KnowledgeEvent.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


async def event_evidence_counts(session) -> dict:
    rows = await session.execute(
        select(EvidenceUnit.event_id, func.count())
        .where(EvidenceUnit.event_id.is_not(None))
        .group_by(EvidenceUnit.event_id)
    )
    return {event_id: count for event_id, count in rows}


async def event_relationship_counts(session, event_ids: list) -> dict:
    """Relationships whose endpoints both belong to evidence units of these events."""
    if not event_ids:
        return {}
    unit_rows = await session.execute(
        select(EvidenceUnit.id, EvidenceUnit.event_id).where(EvidenceUnit.event_id.in_(event_ids))
    )
    unit_to_event = {unit_id: event_id for unit_id, event_id in unit_rows}
    if not unit_to_event:
        return {}
    rel_rows = await session.execute(select(Relationship.source_id, Relationship.target_id))
    counts: Counter = Counter()
    for source_id, target_id in rel_rows:
        event_id = unit_to_event.get(source_id) or unit_to_event.get(target_id)
        if event_id:
            counts[event_id] += 1
    return dict(counts)


async def list_evidence_units(session, limit: int = 200) -> list[EvidenceUnit]:
    result = await session.execute(
        select(EvidenceUnit).order_by(EvidenceUnit.extracted_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


async def get_source_session_map(session, source_ids: list) -> dict:
    """source_id -> {id, title} for whichever chat session each source was
    uploaded into. A source ingested before chat sessions existed (or via
    some other path) simply won't have an entry, which callers treat as
    "ungrouped" rather than an error."""
    if not source_ids:
        return {}
    result = await session.execute(
        select(SessionSource.source_id, ChatSession.id, ChatSession.title)
        .join(ChatSession, ChatSession.id == SessionSource.session_id)
        .where(SessionSource.source_id.in_(source_ids))
    )
    return {row[0]: {"id": str(row[1]), "title": row[2]} for row in result}


async def get_knowledge_event(session, event_id) -> KnowledgeEvent | None:
    return (await session.execute(select(KnowledgeEvent).where(KnowledgeEvent.id == event_id))).scalar_one_or_none()


async def get_evidence_units_by_event(session, event_id) -> list[EvidenceUnit]:
    result = await session.execute(select(EvidenceUnit).where(EvidenceUnit.event_id == event_id))
    return list(result.scalars().all())


async def get_entities_by_ids(session, entity_ids: list) -> list[Entity]:
    if not entity_ids:
        return []
    result = await session.execute(select(Entity).where(Entity.id.in_(entity_ids)))
    return list(result.scalars().all())


async def get_relationships_for_evidence_ids(session, evidence_ids: list) -> list[Relationship]:
    if not evidence_ids:
        return []
    result = await session.execute(
        select(Relationship).where(
            Relationship.source_id.in_(evidence_ids) | Relationship.target_id.in_(evidence_ids)
        )
    )
    return list(result.scalars().all())


async def corpus_stats(session) -> dict:
    """Aggregate live counts + modality distribution + total media duration.

    No fabricated numbers: every field here is a direct count/sum over the
    actual EvidenceUnit/KnowledgeEvent/Relationship tables.
    """
    evidence_count = (await session.execute(select(func.count()).select_from(EvidenceUnit))).scalar_one()
    event_count = (await session.execute(select(func.count()).select_from(KnowledgeEvent))).scalar_one()
    relationship_count = (await session.execute(select(func.count()).select_from(Relationship))).scalar_one()
    source_count = (await session.execute(select(func.count(func.distinct(EvidenceUnit.source_id))))).scalar_one()

    modality_rows = await session.execute(
        select(EvidenceUnit.modality, func.count()).group_by(EvidenceUnit.modality)
    )
    modality_distribution = {modality: count for modality, count in modality_rows}

    duration_rows = await session.execute(
        select(EvidenceUnit.source_id, EvidenceUnit.source_type, func.max(EvidenceUnit.timestamp_end))
        .where(EvidenceUnit.source_type.in_(("video", "audio")))
        .group_by(EvidenceUnit.source_id, EvidenceUnit.source_type)
    )
    total_media_seconds = sum((row[2] or 0.0) for row in duration_rows)

    return {
        "evidence_units": evidence_count,
        "knowledge_events": event_count,
        "relationships": relationship_count,
        "sources": source_count,
        "modality_distribution": modality_distribution,
        "total_media_seconds": total_media_seconds,
    }


async def list_sources(session) -> list[dict]:
    """One row per distinct source_id: type, evidence count, first-seen time, representative locator."""
    rows = await session.execute(
        select(
            EvidenceUnit.source_id, EvidenceUnit.source_type,
            func.count().label("evidence_count"),
            func.min(EvidenceUnit.extracted_at).label("first_seen"),
            func.max(EvidenceUnit.timestamp_end).label("duration_seconds"),
        ).group_by(EvidenceUnit.source_id, EvidenceUnit.source_type)
        .order_by(func.min(EvidenceUnit.extracted_at).desc())
    )
    sources = []
    for source_id, source_type, evidence_count, first_seen, duration_seconds in rows:
        locator_result = await session.execute(
            select(EvidenceUnit.raw_locator)
            .where(EvidenceUnit.source_id == source_id)
            .limit(1)
        )
        raw_locator = locator_result.scalar_one_or_none() or ""
        sources.append({
            "source_id": str(source_id),
            "source_type": source_type,
            "evidence_count": evidence_count,
            "first_seen": first_seen.isoformat() if first_seen else None,
            "duration_seconds": duration_seconds,
            "raw_locator": raw_locator,
        })
    return sources


async def delete_source(session, source_id: uuid.UUID) -> dict:
    """Removes a source and everything derived from it: its evidence units,
    the knowledge events those units built (unless another source's evidence
    still references them), any relationships involving that evidence, and
    the session links pointing at it. Returns the ids so the caller can also
    clean up the media file and the graph."""
    unit_rows = await session.execute(
        select(EvidenceUnit.id, EvidenceUnit.event_id, EvidenceUnit.raw_locator)
        .where(EvidenceUnit.source_id == source_id)
    )
    unit_ids, event_ids, raw_locator = [], set(), None
    for unit_id, event_id, locator in unit_rows:
        unit_ids.append(unit_id)
        if raw_locator is None:
            raw_locator = locator
        if event_id is not None:
            event_ids.add(event_id)

    if unit_ids:
        await session.execute(
            delete(Relationship).where(
                Relationship.source_id.in_(unit_ids) | Relationship.target_id.in_(unit_ids)
            )
        )
        await session.execute(delete(EvidenceUnit).where(EvidenceUnit.source_id == source_id))

    orphaned_events = []
    for event_id in event_ids:
        still_used = await session.execute(
            select(EvidenceUnit.id).where(EvidenceUnit.event_id == event_id).limit(1)
        )
        if still_used.scalar_one_or_none() is None:
            orphaned_events.append(event_id)
    if orphaned_events:
        await session.execute(delete(Relationship).where(
            Relationship.source_id.in_(orphaned_events) | Relationship.target_id.in_(orphaned_events)
        ))
        await session.execute(delete(KnowledgeEvent).where(KnowledgeEvent.id.in_(orphaned_events)))

    await session.execute(delete(SessionSource).where(SessionSource.source_id == source_id))
    await session.commit()
    return {"evidence_unit_ids": unit_ids, "event_ids": list(event_ids), "raw_locator": raw_locator}


async def top_entities(session, limit: int = 10) -> list[dict]:
    """Entities ranked by how many EvidenceUnits actually mention them (real mention count, not a fabricated score)."""
    entities = await list_entities(session)
    units = await session.execute(select(EvidenceUnit.entities))
    mention_counts: Counter = Counter()
    for (entity_ids,) in units:
        for entity_id in entity_ids or []:
            mention_counts[entity_id] += 1
    ranked = sorted(entities, key=lambda e: mention_counts.get(e.id, 0), reverse=True)
    return [
        {
            "id": str(e.id), "canonical_name": e.canonical_name, "entity_type": e.entity_type,
            "mention_count": mention_counts.get(e.id, 0),
        }
        for e in ranked[:limit] if mention_counts.get(e.id, 0) > 0
    ]
