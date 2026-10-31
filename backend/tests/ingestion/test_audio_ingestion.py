import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.ingestion.audio import ingest_audio
from app.extraction.asr import TranscriptSegment
from app.extraction.diarization import SpeakerTurn

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_ingest_audio_creates_speech_units():
    source_id = uuid.uuid4()
    with (
        patch("app.ingestion.audio.transcribe", return_value=[TranscriptSegment(0.0, 3.0, "Meeting starts at 10am", -0.02, 0.0, 0.95)]),
        patch("app.ingestion.audio.diarize", return_value=[SpeakerTurn(0.0, 3.0, "SPEAKER_00")]),
        patch("app.ingestion.audio.embed_texts", return_value=[[0.1] * 1536]),
    ):
        async with get_session() as session:
            units = await ingest_audio(session, "dummy.wav", source_id)
            await session.commit()
    assert len(units) == 1
    assert units[0].modality == "speech"
    assert units[0].source_type == "audio"
