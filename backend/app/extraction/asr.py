import asyncio
import os
import subprocess
from dataclasses import dataclass
from openai import OpenAI
from app.concurrency import call_with_backoff

# OpenAI's whisper-1 endpoint hard-rejects uploads over 25MB. A 16kHz mono
# 16-bit WAV (what extract_audio produces) runs ~32KB/s, so anything past
# ~13 minutes of audio exceeds that -- previously the whole file was
# uploaded in one call regardless of length, which silently failed the
# entire transcription once duration crossed that line (visible as
# "processing stops partway through" on long videos, since the oversized
# upload still takes real time before the server rejects it).
CHUNK_SECONDS = 600.0


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    confidence: float
    speaker: str | None = None


def _call_whisper(audio_path: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(audio_path, "rb") as f:
        return client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="verbose_json",
            timestamp_granularities=["segment"],
        )


def _confidence(avg_logprob: float, no_speech_prob: float) -> float:
    base = max(0.0, min(1.0, 1.0 + avg_logprob))
    return base * (1.0 - no_speech_prob)


def _audio_duration_seconds(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _split_into_chunks(audio_path: str, chunk_seconds: float) -> list[tuple[str, float]]:
    """Returns [(chunk_path, offset_seconds), ...]. A single-chunk file is
    returned as-is (no re-encode) so short audio pays zero extra cost."""
    duration = _audio_duration_seconds(audio_path)
    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]

    out_dir = os.path.dirname(audio_path)
    chunks = []
    offset = 0.0
    i = 0
    while offset < duration:
        chunk_path = os.path.join(out_dir, f"_asr_chunk_{i:03d}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(offset), "-t", str(chunk_seconds), "-i", audio_path, chunk_path],
            check=True, capture_output=True,
        )
        chunks.append((chunk_path, offset))
        offset += chunk_seconds
        i += 1
    return chunks


async def _transcribe_chunk(chunk_path: str, offset: float) -> list[TranscriptSegment]:
    response = await call_with_backoff(_call_whisper, chunk_path)
    return [
        TranscriptSegment(
            start=s.start + offset, end=s.end + offset, text=s.text,
            avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
            confidence=_confidence(s.avg_logprob, s.no_speech_prob),
        )
        for s in response.segments
    ]


async def transcribe(audio_path: str) -> list[TranscriptSegment]:
    chunks = _split_into_chunks(audio_path, CHUNK_SECONDS)
    # Independent chunks transcribe concurrently (bounded by LLM_SEMAPHORE),
    # so a long video pays for its slowest chunk's latency, not the sum.
    results = await asyncio.gather(*(_transcribe_chunk(path, offset) for path, offset in chunks))
    segments = [seg for chunk_segments in results for seg in chunk_segments]
    segments.sort(key=lambda s: s.start)
    return segments
