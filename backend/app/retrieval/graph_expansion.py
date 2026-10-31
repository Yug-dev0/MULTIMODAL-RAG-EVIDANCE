from app.extraction.ner import extract_entities
from app.linking.entity_resolution import resolve_entity


async def entities_for_query(session, query_text: str) -> list:
    mentions = await extract_entities(query_text)
    # Sequential on purpose: each resolve_entity call reads/writes the same
    # AsyncSession, and a single AsyncSession cannot be used concurrently
    # from multiple coroutines. Query mentions are typically few, so this
    # isn't a meaningful cost once extract_entities itself is non-blocking.
    return [await resolve_entity(session, m.text, m.entity_type) for m in mentions]


async def evidence_linked_to_entities(driver, entity_ids: list) -> list[str]:
    async with driver.session() as session:
        result = await session.run(
            "MATCH (v:EvidenceUnit)-[:MENTIONS]->(e:Entity) WHERE e.id IN $ids RETURN DISTINCT v.id AS id",
            ids=[str(i) for i in entity_ids],
        )
        return [record["id"] async for record in result]


async def graph_expand(driver, seed_ids: list[str], hops: int = 2) -> list[str]:
    async with driver.session() as session:
        result = await session.run(
            f"MATCH (start) WHERE start.id IN $ids "
            f"MATCH (start)-[*1..{hops}]-(connected) "
            "RETURN DISTINCT connected.id AS id",
            ids=[str(i) for i in seed_ids],
        )
        return [record["id"] async for record in result]
