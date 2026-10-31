import json
import pytest
from unittest.mock import patch, MagicMock
from app.extraction.ner import extract_entities, EntityMention

pytestmark = pytest.mark.asyncio


async def test_extract_entities_parses_structured_response():
    fake_message = MagicMock()
    fake_message.content = json.dumps({"entities": [
        {"text": "Redis", "entity_type": "technology"},
        {"text": "Rahul", "entity_type": "person"},
    ]})
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    with patch("app.extraction.ner._call_ner_model", return_value=fake_completion):
        entities = await extract_entities("Rahul proposed using Redis for caching.")
    assert entities == [EntityMention("Redis", "technology"), EntityMention("Rahul", "person")]
