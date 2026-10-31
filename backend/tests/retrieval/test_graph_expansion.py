import uuid
import pytest
from app.storage.graph import get_driver, setup_constraints, upsert_entity_node, upsert_evidence_node
from app.retrieval.graph_expansion import evidence_linked_to_entities, graph_expand

pytestmark = pytest.mark.asyncio


async def test_evidence_linked_to_entities_and_graph_expand():
    driver = get_driver()
    await setup_constraints(driver)
    entity_id = str(uuid.uuid4())
    evidence_id = str(uuid.uuid4())
    other_id = str(uuid.uuid4())
    async with driver.session() as session:
        await session.execute_write(upsert_entity_node, {
            "id": entity_id, "canonical_name": "Redis", "entity_type": "technology",
            "description": None, "aliases": [],
        })
        await session.execute_write(upsert_evidence_node, {
            "id": evidence_id, "content_text": "x", "modality": "speech", "source_id": "s",
            "timestamp_start": 0.0, "timestamp_end": 1.0, "confidence": 0.9,
        })
        await session.execute_write(upsert_evidence_node, {
            "id": other_id, "content_text": "y", "modality": "ocr", "source_id": "s",
            "timestamp_start": 2.0, "timestamp_end": 2.0, "confidence": 0.8,
        })
        await session.run(
            "MATCH (v:EvidenceUnit {id:$vid}), (e:Entity {id:$eid}) MERGE (v)-[:MENTIONS]->(e)",
            vid=evidence_id, eid=entity_id,
        )
        await session.run(
            "MATCH (a:EvidenceUnit {id:$a}), (b:EvidenceUnit {id:$b}) MERGE (a)-[:RELATES {relationship_type:'explains'}]->(b)",
            a=evidence_id, b=other_id,
        )

    linked = await evidence_linked_to_entities(driver, [entity_id])
    assert evidence_id in linked

    expanded = await graph_expand(driver, [evidence_id], hops=1)
    assert other_id in expanded
    await driver.close()
