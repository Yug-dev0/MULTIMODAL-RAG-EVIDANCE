from app.storage.graph import upsert_entity_node, upsert_evidence_node, upsert_event_node, upsert_relationship_edge
from app.storage.repository import model_to_dict


async def sync_to_graph(driver, *, entities: list, evidence_units: list, events: list, relationships: list) -> None:
    async with driver.session() as session:
        for entity in entities:
            await session.execute_write(upsert_entity_node, model_to_dict(entity))
        for event in events:
            await session.execute_write(upsert_event_node, model_to_dict(event))
        for unit in evidence_units:
            await session.execute_write(upsert_evidence_node, model_to_dict(unit))
        for rel in relationships:
            await session.execute_write(upsert_relationship_edge, model_to_dict(rel))
        for unit in evidence_units:
            if unit.event_id:
                await session.run(
                    "MATCH (v:EvidenceUnit {id:$vid}), (e:KnowledgeEvent {id:$eid}) MERGE (v)-[:BELONGS_TO]->(e)",
                    vid=str(unit.id), eid=str(unit.event_id),
                )
            for entity_id in unit.entities:
                await session.run(
                    "MATCH (v:EvidenceUnit {id:$vid}), (en:Entity {id:$enid}) MERGE (v)-[:MENTIONS]->(en)",
                    vid=str(unit.id), enid=str(entity_id),
                )
