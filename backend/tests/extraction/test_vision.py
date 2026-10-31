import json
import pytest
from unittest.mock import patch, MagicMock
from app.extraction.vision import caption_image, ImageCaption

pytestmark = pytest.mark.asyncio


async def test_caption_image_parses_structured_response():
    fake_message = MagicMock()
    fake_message.content = json.dumps({"caption": "Architecture diagram", "description": "A diagram showing a load balancer in front of two app servers connected to a Redis cache and a Postgres database.", "confidence": 0.88})
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]
    with patch("app.extraction.vision._call_vision_model", return_value=fake_completion):
        result = await caption_image("dummy/frame.jpg")
    assert isinstance(result, ImageCaption)
    assert result.caption == "Architecture diagram"
    assert "Redis" in result.description
    assert result.confidence == 0.88
