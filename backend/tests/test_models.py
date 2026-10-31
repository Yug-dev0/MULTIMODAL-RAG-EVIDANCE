from app.storage.models import EvidenceUnit, Entity, KnowledgeEvent, Relationship, RelationshipType, EventType

def test_relationship_type_enum_values():
    assert {e.value for e in RelationshipType} == {
        "explains", "supports", "corresponds_to", "mentions", "depicts",
        "contradicts", "follows", "references", "unrelated",
    }

def test_event_type_enum_values():
    assert {e.value for e in EventType} == {
        "discussion", "decision", "proposal", "explanation",
        "diagram_explanation", "definition", "question", "answer",
        "comparison", "action_item", "observation", "reference",
    }

def test_evidence_unit_has_required_columns():
    cols = {c.name for c in EvidenceUnit.__table__.columns}
    assert {
        "id", "content_text", "content_type", "modality", "source_id",
        "source_type", "raw_locator", "extraction_method", "extracted_at",
        "timestamp_start", "timestamp_end", "page_number", "bbox",
        "embedding", "confidence", "confidence_factors", "event_id",
        "metadata_json", "searchable_text", "language",
    }.issubset(cols)
