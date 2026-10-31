import json
import pytest
from unittest.mock import patch, MagicMock
from app.linking.relationship_inference import infer_relationship, RelationshipInference

pytestmark = pytest.mark.asyncio


async def test_infer_relationship_parses_structured_response():
    fake_message = MagicMock()
    fake_message.content = json.dumps({
        "related": True, "relationship_type": "explains", "confidence": 0.93,
        "evidence": "The speaker describes the Redis architecture visible in the frame",
    })
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    with patch("app.linking.relationship_inference._call_relationship_model", return_value=fake_completion):
        result = await infer_relationship("we use Redis for caching", "REDIS ARCHITECTURE DIAGRAM", "ocr")
    assert isinstance(result, RelationshipInference)
    assert result.related is True
    assert result.relationship_type == "explains"
