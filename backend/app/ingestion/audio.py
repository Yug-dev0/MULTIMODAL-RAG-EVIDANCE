import asyncio
import uuid
from app.extraction.asr import transcribe
from app.extraction.diarization import diarize, merge_speakers
from app.extraction.embeddings import embed_texts
from app.storage.repository import persist_evidence_unit


async def ingest_audio(session, audio_path: str, source_id: uuid.UUID) -> list:
    # transcribe is network-bound (OpenAI); diarize is local CPU inference --
    # independent, so run them concurrently instead of paying for both in
    # sequence.
    transcript_segments, turns = await asyncio.gather(
        transcribe(audio_path), asyncio.to_thread(diarize, audio_path),
    )
    # See video.py: Whisper emits empty-text segments for non-speech spans,
    # and embedding an empty string 400s the whole job. Dropping them here
    # keeps the batch below aligned with the segments actually persisted.
    segments = [seg for seg in merge_speakers(transcript_segments, turns) if seg.text.strip()]

    # One batched embeddings call for every segment's text, instead of one
    # call per segment inside the loop below -- for a typical few-minute
    # clip (15-30 short segments) this alone turns ~15-30 sequential API
    # round-trips into 1.
    embeddings = await embed_texts([seg.text for seg in segments])

    units = []
    for seg, embedding in zip(segments, embeddings):
        unit = await persist_evidence_unit(
            session, content_text=seg.text, content_type="transcript", modality="speech",
            source_id=source_id, source_type="audio", raw_locator=f"{audio_path}#t={seg.start},{seg.end}",
            extraction_method="whisper-1+pyannote", timestamp_start=seg.start, timestamp_end=seg.end,
            confidence=seg.confidence,
            confidence_factors={"avg_logprob": seg.avg_logprob, "no_speech_prob": seg.no_speech_prob},
            metadata_json={"speaker": seg.speaker}, embedding=embedding,
        )
        units.append(unit)
    return units
