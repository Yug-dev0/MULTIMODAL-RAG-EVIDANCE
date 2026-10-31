"""
Canonical structured representation for every extracted observation,
regardless of source modality. This is the single contract that ingestion
adapters produce and that storage/retrieval layers consume.

Design intent (see docs/superpowers/specs/ for full rationale):
- A transcript sentence, a video keyframe, an OCR text block, and a PDF
  paragraph are all instances of the same schema, not bespoke per-modality
  records. This is what lets cross-modal linking and retrieval treat them
  uniformly while still preserving full provenance back to the original
  source.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TRANSCRIPT_SEGMENT = "transcript_segment"
    VIDEO_FRAME = "video_frame"
    OCR_TEXT = "ocr_text"
    IMAGE = "image"
    PDF_TEXT = "pdf_text"
    PDF_IMAGE = "pdf_image"


class SourceType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"


class EmbeddingSpace(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    NONE = "none"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Entity(BaseModel):
    text: str
    type: str  # open-vocabulary, LLM-assigned (Person, System, Component, Technology, Concept, Organization, ...)
    confidence: float = Field(ge=0.0, le=1.0)


class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0)


class Provenance(BaseModel):
    extractor: str  # e.g. "faster-whisper", "tesseract", "gemini-2.5-flash", "pymupdf"
    model: Optional[str] = None
    version: Optional[str] = None
    raw_ref: Optional[str] = None  # path to raw artifact this was derived from (frame png, audio segment, etc.)


class CrossModalLink(BaseModel):
    target_evidence_id: str
    link_type: str  # "temporal_cooccurrence" | "semantic_similarity" | "same_entity"
    confidence: float = Field(ge=0.0, le=1.0)
    # Bounded LLM call on top-scored candidates only (see cross_modal_linker.py) —
    # constrained vocabulary so retrieval can reason about *why* two pieces of
    # evidence are connected, not just that they are.
    semantic_label: Optional[str] = None  # "explains"|"depicts"|"supports"|"contradicts"|"follows"|"references"
    details: Optional[str] = None


class EvidenceUnit(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    modality: Modality
    content: str

    source_file: str
    source_type: SourceType

    start_time: Optional[float] = None  # seconds into source (video/audio)
    end_time: Optional[float] = None
    page_number: Optional[int] = None  # PDF positional provenance
    bbox: Optional[BoundingBox] = None  # region provenance (OCR box, detected diagram region)
    speaker: Optional[str] = None

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    provenance: Provenance

    embedding_id: Optional[str] = None
    embedding_space: EmbeddingSpace = EmbeddingSpace.NONE

    cross_modal_links: list[CrossModalLink] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def temporal_range(self) -> Optional[tuple[float, float]]:
        if self.start_time is None:
            return None
        return (self.start_time, self.end_time if self.end_time is not None else self.start_time)
