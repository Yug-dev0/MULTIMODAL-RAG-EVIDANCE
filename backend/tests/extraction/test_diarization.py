from unittest.mock import patch, MagicMock
from app.extraction.asr import TranscriptSegment
from app.extraction.diarization import diarize, merge_speakers, SpeakerTurn


def test_diarize_returns_speaker_turns():
    fake_annotation = MagicMock()
    fake_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=3.0), None, "SPEAKER_00"),
        (MagicMock(start=3.0, end=6.0), None, "SPEAKER_01"),
    ]
    with patch("app.extraction.diarization._run_pipeline", return_value=fake_annotation):
        turns = diarize("dummy/path.wav")
    assert turns == [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(3.0, 6.0, "SPEAKER_01")]


def test_merge_speakers_assigns_by_max_overlap():
    segments = [TranscriptSegment(0.5, 2.5, "hi", -0.05, 0.01, 0.9)]
    turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(3.0, 6.0, "SPEAKER_01")]
    merged = merge_speakers(segments, turns)
    assert merged[0].speaker == "SPEAKER_00"
