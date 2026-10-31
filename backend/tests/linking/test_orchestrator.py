import uuid
import pytest
from unittest.mock import patch, AsyncMock
from app.linking.orchestrator import link_source
from app.storage.models import EvidenceUnit, Relationship

pytestmark = pytest.mark.asyncio


def _unit(text, modality, start, end, embedding, source_type="video"):
    return EvidenceUnit(
        id=uuid.uuid4(), content_text=text, content_type="x", modality=modality,
        source_id=uuid.uuid4(), source_type=source_type, raw_locator="v", extraction_method="x",
        timestamp_start=start, timestamp_end=end, embedding=embedding, confidence=0.9,
        entities=[], metadata_json={},
    )


async def test_link_source_creates_multimodal_event_from_related_speech_and_frame():
    speech = _unit("we should use redis for caching", "speech", 0.0, 5.0, [1.0, 0.0])
    frame = _unit("REDIS ARCHITECTURE", "ocr", 2.0, 2.0, [0.98, 0.02])

    from app.linking.relationship_inference import RelationshipInference
    from app.linking.summarization import EventSummary
    from app.storage.models import KnowledgeEvent

    fake_event = KnowledgeEvent(id=uuid.uuid4(), event_type="diagram_explanation", title="Redis caching",
                                 summary="Redis proposed for caching", timestamp_start=0.0, timestamp_end=5.0,
                                 confidence=0.9, entities=[], metadata_json={})
    fake_relationship = Relationship(id=uuid.uuid4(), source_type="evidence", source_id=speech.id,
                                      target_type="evidence", target_id=frame.id, relationship_type="explains",
                                      confidence=0.9, evidence_ids=[speech.id, frame.id], metadata_json={})

    with (
        patch("app.linking.orchestrator.get_evidence_units_by_source", new=AsyncMock(return_value=[speech, frame])),
        patch("app.linking.orchestrator.infer_relationship", new=AsyncMock(
            return_value=RelationshipInference(True, "explains", 0.9, "matches"))),
        patch("app.linking.orchestrator.extract_entities", return_value=[]),
        patch("app.linking.orchestrator.summarize_event", new=AsyncMock(
            return_value=EventSummary("diagram_explanation", "Redis caching", "Redis proposed for caching"))),
        patch("app.linking.orchestrator.assemble_knowledge_event", new=AsyncMock(return_value=fake_event)),
        patch("app.linking.orchestrator.persist_relationship", new=AsyncMock(return_value=fake_relationship)) as mock_persist_rel,
        patch("app.linking.orchestrator.detect_and_record_contradictions", new=AsyncMock(return_value=[])),
        patch("app.linking.orchestrator.sync_to_graph", new=AsyncMock()) as mock_sync,
    ):
        result = await link_source(None, None, uuid.uuid4())

    assert result["events_created"] >= 1
    assert result["relationships_created"] >= 1
    mock_persist_rel.assert_awaited_once()
    _, kwargs = mock_persist_rel.call_args
    assert kwargs["relationship_type"] == "explains"
    assert kwargs["source_id"] == speech.id and kwargs["target_id"] == frame.id
    # the persisted relationship must actually reach the graph, not be discarded
    _, sync_kwargs = mock_sync.call_args
    assert fake_relationship in sync_kwargs["relationships"]


async def test_link_source_passes_resolved_entities_to_graph_sync():
    # Regression test: sync_to_graph used to always receive entities=[]
    # (a hardcoded empty list), so Entity nodes never reached Neo4j even
    # though they were correctly persisted to Postgres.
    from app.extraction.ner import EntityMention
    from app.storage.models import Entity
    from app.linking.summarization import EventSummary
    from app.storage.models import KnowledgeEvent

    speech = _unit("we should use redis for caching", "speech", 0.0, 5.0, [1.0, 0.0])
    redis_entity = Entity(id=uuid.uuid4(), canonical_name="Redis", entity_type="technology", aliases=[])
    fake_event = KnowledgeEvent(id=uuid.uuid4(), event_type="discussion", title="t", summary="s",
                                 timestamp_start=0.0, timestamp_end=5.0, confidence=0.9, entities=[], metadata_json={})

    with (
        patch("app.linking.orchestrator.get_evidence_units_by_source", new=AsyncMock(return_value=[speech])),
        patch("app.linking.orchestrator.extract_entities", return_value=[EntityMention("Redis", "technology")]),
        patch("app.linking.orchestrator.resolve_entity", new=AsyncMock(return_value=redis_entity)),
        patch("app.linking.orchestrator.summarize_event", new=AsyncMock(
            return_value=EventSummary("discussion", "t", "s"))),
        patch("app.linking.orchestrator.assemble_knowledge_event", new=AsyncMock(return_value=fake_event)),
        patch("app.linking.orchestrator.detect_and_record_contradictions", new=AsyncMock(return_value=[])),
        patch("app.linking.orchestrator.sync_to_graph", new=AsyncMock()) as mock_sync,
    ):
        await link_source(None, None, uuid.uuid4())

    _, kwargs = mock_sync.call_args
    assert redis_entity in kwargs["entities"]


async def test_link_source_reports_pipeline_stages_in_order():
    speech = _unit("we should use redis for caching", "speech", 0.0, 5.0, [1.0, 0.0])
    stages_seen = []

    async def record_stage(stage):
        stages_seen.append(stage)

    with (
        patch("app.linking.orchestrator.get_evidence_units_by_source", new=AsyncMock(return_value=[speech])),
        patch("app.linking.orchestrator.extract_entities", return_value=[]),
        patch("app.linking.orchestrator.summarize_event", new=AsyncMock(
            side_effect=Exception("no candidates to summarize in this test"))),
        patch("app.linking.orchestrator.assemble_knowledge_event", new=AsyncMock(return_value=None)),
        patch("app.linking.orchestrator.sync_to_graph", new=AsyncMock()),
    ):
        try:
            await link_source(None, None, uuid.uuid4(), on_stage=record_stage)
        except Exception:
            pass

    assert stages_seen[0] == "understand"
    assert "link" in stages_seen
