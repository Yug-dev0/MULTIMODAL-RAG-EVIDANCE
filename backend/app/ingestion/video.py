import asyncio
import os
import uuid
from app.extraction.video import extract_audio, extract_keyframes
from app.extraction.asr import transcribe
from app.extraction.diarization import diarize, merge_speakers
from app.extraction.ocr import ocr_image
from app.extraction.vision import caption_image
from app.extraction.embeddings import embed_texts
from app.storage.repository import persist_evidence_unit


async def ingest_video(session, video_path: str, source_id: uuid.UUID, media_dir: str) -> list:
    work_dir = os.path.join(media_dir, str(source_id))
    # extract_audio and extract_keyframes are independent, blocking
    # subprocess work (ffmpeg / scene detection) -- running them on threads
    # concurrently overlaps two CPU-bound steps that previously ran back to
    # back for the whole length of the video.
    audio_path, keyframes = await asyncio.gather(
        asyncio.to_thread(extract_audio, video_path, work_dir),
        asyncio.to_thread(extract_keyframes, video_path, work_dir),
    )

    # transcribe is network-bound (OpenAI, chunked+parallel internally for
    # long audio); diarize is local CPU inference -- independent of each
    # other too.
    transcript_segments, turns = await asyncio.gather(
        transcribe(audio_path), asyncio.to_thread(diarize, audio_path),
    )
    # Whisper emits a segment for every span it processes, including
    # silent/non-speech stretches (music, tone, background noise) -- those
    # come back with empty or whitespace-only text, which the embeddings API
    # then rejects outright, failing the whole job. Drop them before
    # embedding rather than skipping mid-loop, so the batch below stays
    # aligned with the segments actually persisted.
    segments = [seg for seg in merge_speakers(transcript_segments, turns) if seg.text.strip()]

    # One batched embeddings call for every segment's text instead of one
    # call per segment inside the loop below -- a several-minute video can
    # easily have 20-30+ segments, turning that many sequential API
    # round-trips into 1.
    speech_embeddings = await embed_texts([seg.text for seg in segments])

    units = []
    for seg, embedding in zip(segments, speech_embeddings):
        unit = await persist_evidence_unit(
            session, content_text=seg.text, content_type="transcript", modality="speech",
            source_id=source_id, source_type="video", raw_locator=f"{video_path}#t={seg.start},{seg.end}",
            extraction_method="whisper-1+pyannote", timestamp_start=seg.start, timestamp_end=seg.end,
            confidence=seg.confidence,
            confidence_factors={"avg_logprob": seg.avg_logprob, "no_speech_prob": seg.no_speech_prob},
            metadata_json={"speaker": seg.speaker}, embedding=embedding,
        )
        units.append(unit)

    # Captioning is the slow part (one GPT-4o-mini vision call per keyframe) —
    # firing them all concurrently (bounded by LLM_SEMAPHORE inside
    # caption_image) collapses N sequential round-trips into ~N/5. OCR is
    # local/fast, so it stays a plain sequential loop.
    captions = await asyncio.gather(*(caption_image(frame.image_path) for frame in keyframes))
    ocr_results = [ocr_image(frame.image_path) for frame in keyframes]

    # Batch-embed OCR text and captions in (at most) two calls total instead
    # of up to two per frame. Only frames with substantive OCR text get an
    # OCR unit at all, so track which indices those are to map embeddings
    # back to the right frame after the batched call.
    ocr_indices = [i for i, (full_text, _mean_conf, _words) in enumerate(ocr_results) if full_text.strip()]
    ocr_embeddings, caption_embeddings = await asyncio.gather(
        embed_texts([ocr_results[i][0] for i in ocr_indices]),
        embed_texts([c.description for c in captions]),
    )
    ocr_embedding_by_index = dict(zip(ocr_indices, ocr_embeddings))

    for i, (frame, caption) in enumerate(zip(keyframes, captions)):
        full_text, mean_conf, _words = ocr_results[i]
        if full_text.strip():
            unit = await persist_evidence_unit(
                session, content_text=full_text, content_type="ocr", modality="ocr",
                source_id=source_id, source_type="video", raw_locator=frame.image_path,
                extraction_method="tesseract", timestamp_start=frame.timestamp,
                timestamp_end=frame.timestamp, confidence=mean_conf, embedding=ocr_embedding_by_index[i],
            )
            units.append(unit)

        unit = await persist_evidence_unit(
            session, content_text=caption.description, content_type="caption", modality="image",
            source_id=source_id, source_type="video", raw_locator=frame.image_path,
            extraction_method="gpt-4o-mini-vision", timestamp_start=frame.timestamp,
            timestamp_end=frame.timestamp, confidence=caption.confidence,
            metadata_json={"caption": caption.caption}, embedding=caption_embeddings[i],
        )
        units.append(unit)

    return units
