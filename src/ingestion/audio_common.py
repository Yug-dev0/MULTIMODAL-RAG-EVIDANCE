"""
Shared ASR helper used by both the audio adapter (standalone audio files)
and the video adapter (audio track extracted from video), so transcription
+ confidence scoring is implemented once.
"""

from __future__ import annotations

import logging
import math

import config

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper model %s on %s", config.WHISPER_MODEL_SIZE, config.WHISPER_DEVICE)
        _model = WhisperModel(
            config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE
        )
    return _model


def transcribe(audio_path: str) -> list[dict]:
    """
    Returns a list of {"text", "start", "end", "confidence"} segments,
    following Whisper's natural speech-pause boundaries rather than
    fixed-size windows. `confidence` in [0,1] combines the model's average
    per-token log-probability with its no-speech probability for the
    segment — both real, per-segment signals, not a constant.
    """
    model = _get_model()
    segments, _info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

    out = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        prob = math.exp(seg.avg_logprob)  # rough per-token geometric-mean probability
        confidence = max(0.0, min(1.0, prob * (1.0 - seg.no_speech_prob)))
        out.append({"text": text, "start": float(seg.start), "end": float(seg.end), "confidence": confidence})
    return out
