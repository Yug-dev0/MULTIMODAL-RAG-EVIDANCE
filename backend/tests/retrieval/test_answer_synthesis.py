import json
import uuid
import pytest
from unittest.mock import patch, MagicMock
from app.storage.models import EvidenceUnit, KnowledgeEvent
from app.retrieval.answer_synthesis import synthesize_answer, AnswerResult

pytestmark = pytest.mark.asyncio


def _evidence(modality, text, confidence=0.9):
    return EvidenceUnit(
        id=uuid.uuid4(), content_text=text, content_type="x", modality=modality,
        source_id=uuid.uuid4(), source_type="video", raw_locator="frame.jpg",
        extraction_method="x", timestamp_start=2.0, timestamp_end=2.0, confidence=confidence,
        entities=[], metadata_json={},
    )


async def test_synthesize_answer_builds_citations_from_evidence_not_model():
    speech = _evidence("speech", "We propose Redis to reduce database load")
    frame = _evidence("ocr", "REDIS ARCHITECTURE DIAGRAM")
    event = KnowledgeEvent(id=uuid.uuid4(), event_type="diagram_explanation", title="t", summary="s",
                            timestamp_start=0.0, timestamp_end=5.0, confidence=0.9, entities=[], metadata_json={})

    fake_message = MagicMock()
    fake_message.content = json.dumps({
        "answer": "Redis was proposed to reduce database load, shown in the accompanying architecture diagram.",
        "cited_evidence_ids": [str(speech.id), str(frame.id)],
    })
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_message)]

    with patch("app.retrieval.answer_synthesis._call_answer_model", return_value=fake_completion):
        result = await synthesize_answer(
            "What architecture was proposed to reduce database load?",
            ranked_evidence=[speech, frame], ranked_events=[event],
        )

    assert isinstance(result, AnswerResult)
    assert len(result.citations) == 2
    assert {c.evidence_id for c in result.citations} == {str(speech.id), str(frame.id)}
    assert result.citations[0].source == "frame.jpg" or result.citations[1].source == "frame.jpg"


async def test_synthesize_answer_refuses_instead_of_hallucinating_when_no_evidence_matched():
    # A session-scoped query with zero evidence in scope (e.g. a chat topic
    # queried before or without matching content) used to still call the LLM
    # with empty evidence/events, and it would answer from general model
    # knowledge instead of admitting it had nothing to go on. Verified live
    # earlier: asking "What is Redis?" with no evidence returned a real
    # definition of Redis instead of "no evidence." This is now a fast path
    # that skips the LLM call entirely and is honest about it.
    with patch("app.retrieval.answer_synthesis._call_answer_model") as mock_call:
        result = await synthesize_answer("What is Redis?", ranked_evidence=[], ranked_events=[])

    mock_call.assert_not_called()
    assert result.answer == "I cannot find the answer in the uploaded files."
    assert result.citations == []
    assert result.events_used == []
