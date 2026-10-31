import pytest
from app.storage.graph import get_driver, setup_constraints, upsert_entity_node

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_entity_roundtrip():
    driver = get_driver()
    await setup_constraints(driver)
    async with driver.session() as session:
        await session.execute_write(upsert_entity_node, {
            "id": "11111111-1111-1111-1111-111111111111",
            "canonical_name": "Redis", "entity_type": "technology",
            "description": None, "aliases": ["redis-cache"],
        })
        result = await session.run(
            "MATCH (e:Entity {id: $id}) RETURN e.canonical_name AS name",
            id="11111111-1111-1111-1111-111111111111",
        )
        record = await result.single()
        assert record["name"] == "Redis"
    await driver.close()
