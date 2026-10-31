import uuid
import pytest
from unittest.mock import patch, AsyncMock
from app.baseline.naive_chunk_rag import chunk_text, run_baseline_query



def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-50:] == second_words[:50]


async def test_run_baseline_query_does_plain_vector_search_and_llm_call():
    from app.storage.models import BaselineChunk
    chunk = BaselineChunk(id=uuid.uuid4(), content_text="Redis reduces load on Postgres.",
                           source_id=uuid.uuid4(), source_type="video", chunk_index=0, metadata_json={})
    with (
        patch("app.baseline.naive_chunk_rag.embed_texts", return_value=[[0.1] * 1536]),
        patch("app.baseline.naive_chunk_rag.baseline_vector_search", new=AsyncMock(return_value=[(chunk, 0.9)])),
        patch("app.baseline.naive_chunk_rag._call_baseline_answer_model") as mock_llm,
    ):
        mock_llm.return_value.choices = [type("M", (), {"message": type("M2", (), {"content": "Redis reduces database load."})()})]
        answer = await run_baseline_query(None, "What reduces database load?")
    assert "Redis" in answer
