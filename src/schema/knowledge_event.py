"""
A KnowledgeEvent materializes a cluster of tightly cross-modal-linked
EvidenceUnits into one retrieval-friendly object (title + summary + time
span), rather than leaving retrieval to reassemble connected evidence from
scratch on every query. See docs/superpowers/specs/2026-08-23-merge-with-
teammate-design-addendum.md for why this exists.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class KnowledgeEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    summary: str
    evidence_ids: list[str]
    modalities: list[str]
    source_files: list[str]
    start_time: float | None = None
    end_time: float | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
