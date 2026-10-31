import logging
from fastapi import APIRouter
from app.storage.graph import get_driver

logger = logging.getLogger(__name__)
router = APIRouter()


async def fetch_graph_snapshot(limit: int = 800) -> dict:
    try:
        driver = get_driver()
        async with driver.session() as session:
            node_result = await session.run(
                "MATCH (n) WHERE n:Entity OR n:EvidenceUnit OR n:KnowledgeEvent "
                "RETURN n.id AS id, "
                "coalesce(n.canonical_name, n.title, left(n.content_text, 60)) AS label, "
                "CASE WHEN n:Entity THEN 'entity' WHEN n:KnowledgeEvent THEN 'event' ELSE 'evidence' END AS type, "
                "n.entity_type AS entity_type, n.aliases AS aliases, n.description AS description, "
                "n.event_type AS event_type, n.summary AS summary, "
                "n.modality AS modality, n.content_text AS content_text, n.source_id AS source_id, "
                "n.timestamp_start AS timestamp_start, n.timestamp_end AS timestamp_end, n.confidence AS confidence "
                "LIMIT $limit",
                limit=limit,
            )
            nodes = [dict(record) async for record in node_result]

            link_result = await session.run(
                "MATCH (a)-[r]->(b) WHERE a.id IS NOT NULL AND b.id IS NOT NULL "
                "RETURN a.id AS source, b.id AS target, "
                "coalesce(r.relationship_type, type(r)) AS relationship_type, r.confidence AS confidence "
                "LIMIT $limit",
                limit=limit,
            )
            links = [dict(record) async for record in link_result]
        await driver.close()
        return {"nodes": nodes, "links": links}
    except Exception as e:
        logger.warning(f"Neo4j graph fetch failed: {e}. Returning empty graph.")
        return {"nodes": [], "links": []}


@router.get("/graph")
async def graph():
    return await fetch_graph_snapshot()
