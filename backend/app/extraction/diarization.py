import logging
import os
from dataclasses import dataclass, replace
from app.extraction.asr import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass(eq=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


_pipeline = None
_pipeline_load_failed = False


def _run_pipeline(audio_path: str):
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=os.environ.get("HF_TOKEN"),
        )
    return _pipeline(audio_path)


def diarize(audio_path: str) -> list[SpeakerTurn]:
    """
    Best-effort speaker diarization. pyannote.audio is a gated HF model with a
    heavy, version-sensitive dependency chain (torch/torchaudio/lightning) —
    if it's unavailable or the environment isn't set up for it, ingestion
    should still succeed with unlabeled speakers rather than fail outright.

    Model-load failures (missing/incompatible deps, bad HF_TOKEN) are
    permanent for this process — every subsequent call would fail
    identically, so we stop retrying. A failure on an already-loaded
    pipeline is treated as specific to that one file and does NOT disable
    diarization for later files.
    """
    global _pipeline_load_failed
    if _pipeline_load_failed:
        return []
    try:
        annotation = _run_pipeline(audio_path)
    except Exception as e:
        if _pipeline is None:
            logger.warning(
                "Speaker diarization model unavailable (%s), disabling for this process", type(e).__name__
            )
            _pipeline_load_failed = True
        else:
            logger.warning("Speaker diarization failed for this file (%s)", type(e).__name__)
        return []
    return [
        SpeakerTurn(start=segment.start, end=segment.end, speaker=label)
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_speakers(segments: list[TranscriptSegment], turns: list[SpeakerTurn]) -> list[TranscriptSegment]:
    merged = []
    for seg in segments:
        best_turn, best_overlap = None, 0.0
        for turn in turns:
            ov = _overlap(seg.start, seg.end, turn.start, turn.end)
            if ov > best_overlap:
                best_turn, best_overlap = turn, ov
        merged.append(replace(seg, speaker=best_turn.speaker if best_turn else None))
    return merged
