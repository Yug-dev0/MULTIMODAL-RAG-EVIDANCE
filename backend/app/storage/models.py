import enum
import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.storage.db import Base

EMBEDDING_DIM = 1536  # text-embedding-3-small


class RelationshipType(str, enum.Enum):
    explains = "explains"
    supports = "supports"
    corresponds_to = "corresponds_to"
    mentions = "mentions"
    depicts = "depicts"
    contradicts = "contradicts"
    follows = "follows"
    references = "references"
    unrelated = "unrelated"


class EventType(str, enum.Enum):
    discussion = "discussion"
    decision = "decision"
    proposal = "proposal"
    explanation = "explanation"
    diagram_explanation = "diagram_explanation"
    definition = "definition"
    question = "question"
    answer = "answer"
    comparison = "comparison"
    action_item = "action_item"
    observation = "observation"
    reference = "reference"


def _uuid_col():
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class KnowledgeEvent(Base):
    __tablename__ = "knowledge_events"
    id: Mapped[uuid.UUID] = _uuid_col()
    event_type: Mapped[EventType] = mapped_column(SAEnum(EventType, name="event_type_enum"))
    title: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(String)
    timestamp_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    entities: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_units: Mapped[list["EvidenceUnit"]] = relationship(back_populates="event")


class Entity(Base):
    __tablename__ = "entities"
    id: Mapped[uuid.UUID] = _uuid_col()
    canonical_name: Mapped[str] = mapped_column(String, index=True)
    entity_type: Mapped[str] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class EvidenceUnit(Base):
    __tablename__ = "evidence_units"
    id: Mapped[uuid.UUID] = _uuid_col()
    content_text: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(String)
    modality: Mapped[str] = mapped_column(String)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    source_type: Mapped[str] = mapped_column(String)
    raw_locator: Mapped[str] = mapped_column(String)
    extraction_method: Mapped[str] = mapped_column(String)
    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    timestamp_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    entities: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    confidence: Mapped[float] = mapped_column(Float)
    confidence_factors: Mapped[dict] = mapped_column(JSON, default=dict)
    event_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("knowledge_events.id"), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    searchable_text: Mapped[str] = mapped_column(String)
    language: Mapped[str] = mapped_column(String, default="en")
    event: Mapped[KnowledgeEvent | None] = relationship(back_populates="evidence_units")


class Relationship(Base):
    __tablename__ = "relationships"
    id: Mapped[uuid.UUID] = _uuid_col()
    source_type: Mapped[str] = mapped_column(String)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    target_type: Mapped[str] = mapped_column(String)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    relationship_type: Mapped[RelationshipType] = mapped_column(SAEnum(RelationshipType, name="relationship_type_enum"))
    confidence: Mapped[float] = mapped_column(Float)
    timestamp_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[uuid.UUID] = _uuid_col()
    owner_uid: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String, default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[uuid.UUID] = _uuid_col()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    citations_json: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SessionSource(Base):
    """Maps an ingested source_id to the chat session it was uploaded into.

    Kept as a thin join table (rather than a session_id column on
    EvidenceUnit/KnowledgeEvent) so scoping a topic to its own chat history
    doesn't require threading a new parameter through the whole ingestion
    and event-assembly pipeline -- retrieval just filters by source_id.
    """
    __tablename__ = "session_sources"
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BaselineChunk(Base):
    __tablename__ = "baseline_chunks"
    id: Mapped[uuid.UUID] = _uuid_col()
    content_text: Mapped[str] = mapped_column(String)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    source_type: Mapped[str] = mapped_column(String)
    chunk_index: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
