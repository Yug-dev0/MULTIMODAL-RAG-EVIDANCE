import uuid
import pytest
from unittest.mock import patch, AsyncMock
from app.storage.models import Entity
from app.linking.entity_resolution import resolve_entity

pytestmark = pytest.mark.asyncio


async def test_resolve_entity_matches_by_alias():
    existing = Entity(id=uuid.uuid4(), canonical_name="PostgreSQL", entity_type="technology",
                       aliases=["postgres", "postgres db"], embedding=[1.0, 0.0])
    with patch("app.linking.entity_resolution.list_entities", new=AsyncMock(return_value=[existing])):
        result = await resolve_entity(None, "Postgres DB", "technology")
    assert result.id == existing.id


async def test_resolve_entity_matches_by_embedding_similarity_above_threshold():
    existing = Entity(id=uuid.uuid4(), canonical_name="Redis", entity_type="technology",
                       aliases=[], embedding=[1.0, 0.0])
    with (
        patch("app.linking.entity_resolution.list_entities", new=AsyncMock(return_value=[existing])),
        patch("app.linking.entity_resolution.embed_texts", return_value=[[0.99, 0.01]]),
    ):
        result = await resolve_entity(None, "redis-cache", "technology", threshold=0.9)
    assert result.id == existing.id
    assert "redis-cache" in existing.aliases


async def test_resolve_entity_creates_new_when_no_match():
    new_entity = Entity(id=uuid.uuid4(), canonical_name="Kafka", entity_type="technology", aliases=[])
    with (
        patch("app.linking.entity_resolution.list_entities", new=AsyncMock(return_value=[])),
        patch("app.linking.entity_resolution.embed_texts", return_value=[[0.0, 1.0]]),
        patch("app.linking.entity_resolution.persist_entity", new=AsyncMock(return_value=new_entity)),
    ):
        result = await resolve_entity(None, "Kafka", "technology")
    assert result.canonical_name == "Kafka"
