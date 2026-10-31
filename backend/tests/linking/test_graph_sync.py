import uuid
import pytest
from app.storage.graph import get_driver, setup_constraints
from app.storage.models import Entity, EvidenceUnit, KnowledgeEvent
from app.linking.graph_sync import sync_to_graph

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_sync_to_graph_creates_belongs_to_and_mentions_edges():
    driver = get_driver()
    await setup_constraints(driver)
    entity = Entity(id=uuid.uuid4(), canonical_name="Redis", entity_type="technology",
                     description=None, aliases=[], metadata_json={})
    event = KnowledgeEvent(id=uuid.uuid4(), event_type="explanation", title="t", summary="s",
                            timestamp_start=0.0, timestamp_end=5.0, confidence=0.9, entities=[], metadata_json={})
    unit = EvidenceUnit(id=uuid.uuid4(), content_text="x", content_type="transcript", modality="speech",
                         source_id=uuid.uuid4(), source_type="video", raw_locator="v",
                         extraction_method="whisper-1", timestamp_start=0.0, timestamp_end=1.0,
                         confidence=0.9, entities=[entity.id], event_id=event.id, metadata_json={})

    await sync_to_graph(driver, entities=[entity], evidence_units=[unit], events=[event], relationships=[])

    async with driver.session() as session:
        result = await session.run(
            "MATCH (v:EvidenceUnit {id:$vid})-[:BELONGS_TO]->(e:KnowledgeEvent {id:$eid}) RETURN count(*) AS c",
            vid=str(unit.id), eid=str(event.id),
        )
        assert (await result.single())["c"] == 1
        result = await session.run(
            "MATCH (v:EvidenceUnit {id:$vid})-[:MENTIONS]->(en:Entity {id:$enid}) RETURN count(*) AS c",
            vid=str(unit.id), enid=str(entity.id),
        )
        assert (await result.single())["c"] == 1
    await driver.close()
