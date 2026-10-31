import pytest
from unittest.mock import patch, MagicMock
from app.extraction.embeddings import embed_texts

pytestmark = pytest.mark.asyncio


async def test_embed_texts_returns_vectors_per_input():
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    with patch("app.extraction.embeddings._call_embeddings", return_value=fake_response):
        vectors = await embed_texts(["hello", "world"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
