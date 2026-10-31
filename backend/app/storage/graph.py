import os
from neo4j import AsyncGraphDatabase


def get_driver():
    return AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "ragdb_neo4j")),
    )


async def setup_constraints(driver):
    async with driver.session() as session:
        for label in ("Entity", "EvidenceUnit", "KnowledgeEvent"):
            await session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")


async def upsert_entity_node(tx, entity: dict):
    await tx.run(
        "MERGE (e:Entity {id: $id}) SET e.canonical_name=$canonical_name, "
        "e.entity_type=$entity_type, e.description=$description, e.aliases=$aliases",
        **{k: entity.get(k) for k in ("id", "canonical_name", "entity_type", "description", "aliases")},
    )


async def upsert_evidence_node(tx, evidence: dict):
    await tx.run(
        "MERGE (v:EvidenceUnit {id: $id}) SET v.content_text=$content_text, v.modality=$modality, "
        "v.source_id=$source_id, v.timestamp_start=$timestamp_start, v.timestamp_end=$timestamp_end, "
        "v.confidence=$confidence",
        **{k: evidence.get(k) for k in (
            "id", "content_text", "modality", "source_id", "timestamp_start", "timestamp_end", "confidence"
        )},
    )


async def upsert_event_node(tx, event: dict):
    await tx.run(
        "MERGE (ev:KnowledgeEvent {id: $id}) SET ev.event_type=$event_type, ev.title=$title, "
        "ev.summary=$summary, ev.timestamp_start=$timestamp_start, ev.timestamp_end=$timestamp_end, "
        "ev.confidence=$confidence",
        **{k: event.get(k) for k in (
            "id", "event_type", "title", "summary", "timestamp_start", "timestamp_end", "confidence"
        )},
    )


async def upsert_relationship_edge(tx, rel: dict):
    await tx.run(
        "MATCH (a {id: $source_id}), (b {id: $target_id}) "
        "MERGE (a)-[r:RELATES {relationship_type: $relationship_type}]->(b) "
        "SET r.confidence=$confidence, r.metadata_json=$metadata_json, r.id=$id",
        source_id=str(rel["source_id"]), target_id=str(rel["target_id"]),
        relationship_type=str(rel["relationship_type"]), confidence=rel["confidence"],
        metadata_json=str(rel.get("metadata_json", {})), id=str(rel["id"]),
    )
