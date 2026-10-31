import pytest
from unittest.mock import patch, MagicMock
from app.extraction.asr import transcribe, TranscriptSegment

pytestmark = pytest.mark.asyncio


async def test_transcribe_maps_segments_and_confidence():
    fake_response = MagicMock()
    fake_response.segments = [
        MagicMock(start=0.0, end=2.5, text="Hello world", avg_logprob=-0.1, no_speech_prob=0.02),
    ]
    with (
        patch("app.extraction.asr._audio_duration_seconds", return_value=5.0),
        patch("app.extraction.asr._call_whisper", return_value=fake_response),
    ):
        segments = await transcribe("dummy/path.wav")
    assert len(segments) == 1
    seg = segments[0]
    assert isinstance(seg, TranscriptSegment)
    assert seg.text == "Hello world"
    assert seg.start == 0.0 and seg.end == 2.5
    assert 0.0 <= seg.confidence <= 1.0
    assert seg.speaker is None


async def test_transcribe_chunks_audio_over_the_openai_size_limit_and_offsets_timestamps():
    # Regression test: OpenAI's whisper-1 endpoint rejects uploads over
    # 25MB. A 16kHz mono WAV runs ~32KB/s, so any audio longer than ~13
    # minutes used to be uploaded whole and fail -- this verifies long audio
    # gets split into chunks, each chunk's segments get offset by its
    # position in the full file, and the merged result is time-ordered.
    chunk_1_response = MagicMock()
    chunk_1_response.segments = [MagicMock(start=1.0, end=2.0, text="first chunk", avg_logprob=-0.1, no_speech_prob=0.0)]
    chunk_2_response = MagicMock()
    chunk_2_response.segments = [MagicMock(start=3.0, end=4.0, text="second chunk", avg_logprob=-0.1, no_speech_prob=0.0)]

    with (
        patch("app.extraction.asr._audio_duration_seconds", return_value=900.0),  # 15 min > 10 min chunk size
        patch("app.extraction.asr._split_into_chunks", return_value=[
            ("chunk_000.wav", 0.0), ("chunk_001.wav", 600.0),
        ]),
        patch("app.extraction.asr._call_whisper", side_effect=[chunk_1_response, chunk_2_response]) as mock_call,
    ):
        segments = await transcribe("dummy/long.wav")

    assert mock_call.call_count == 2
    # second chunk's timestamps are offset by its 600s position in the file
    assert [s.text for s in segments] == ["first chunk", "second chunk"]
    assert segments[0].start == 1.0
    assert segments[1].start == 603.0 and segments[1].end == 604.0
