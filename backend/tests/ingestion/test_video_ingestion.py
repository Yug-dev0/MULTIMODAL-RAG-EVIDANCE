import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.ingestion.video import ingest_video
from app.extraction.video import Keyframe
from app.extraction.asr import TranscriptSegment
from app.extraction.diarization import SpeakerTurn
from app.extraction.ocr import OCRWord
from app.extraction.vision import ImageCaption

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_ingest_video_creates_speech_ocr_and_caption_units(tmp_path):
    source_id = uuid.uuid4()
    with (
        patch("app.ingestion.video.extract_audio", return_value=str(tmp_path / "a.wav")),
        patch("app.ingestion.video.extract_keyframes", return_value=[Keyframe(5.0, str(tmp_path / "f0.jpg"))]),
        patch("app.ingestion.video.transcribe", return_value=[TranscriptSegment(0.0, 6.0, "We should use Redis", -0.05, 0.01, 0.9)]),
        patch("app.ingestion.video.diarize", return_value=[SpeakerTurn(0.0, 6.0, "SPEAKER_00")]),
        patch("app.ingestion.video.ocr_image", return_value=("REDIS ARCHITECTURE", 0.85, [OCRWord("REDIS", (0,0,10,10), 0.9)])),
        patch("app.ingestion.video.caption_image", return_value=ImageCaption("Architecture diagram", "A diagram of a Redis cache in front of Postgres.", 0.8)),
        patch("app.ingestion.video.embed_texts", return_value=[[0.1] * 1536]),
    ):
        async with get_session() as session:
            units = await ingest_video(session, "dummy.mp4", source_id, str(tmp_path))
            await session.commit()
    modalities = sorted(u.modality for u in units)
    assert modalities == ["image", "ocr", "speech"]
    speech_unit = next(u for u in units if u.modality == "speech")
    assert speech_unit.metadata_json.get("speaker") == "SPEAKER_00"
