"""
Video ingestion: extracts the audio track and transcribes it, detects visual
scene changes and treats each as a keyframe (region-aware OCR + LLM
captioning), and persists every resulting observation as an EvidenceUnit.
This is the primary adapter for a recorded talk/meeting where speech and an
on-screen diagram need to stay connected on the same timeline.

Entity extraction is batched (see pdf.py's docstring for why): transcript
segments and OCR text units are collected first, then their text is sent
through the LLM provider in a handful of BATCH_SIZE-sized calls rather than
one call per unit.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import config
from src.enrichment.llm_provider import get_llm_provider
from src.ingestion.audio_common import transcribe
from src.ingestion.persistence import enrich_and_persist, resolve_and_assign_entities
from src.ingestion.vision_common import caption_image, is_substantive_ocr, run_ocr_regions
from src.schema.evidence_unit import BoundingBox, EvidenceUnit, Modality, Provenance, SourceType
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

BATCH_SIZE = 150


def _extract_audio(video_path: str, out_dir: Path) -> str:
    audio_path = out_dir / (Path(video_path).stem + ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", str(audio_path)],
        check=True, capture_output=True,
    )
    return str(audio_path)


def _detect_keyframes(video_path: str, frames_dir: Path) -> list[tuple[float, str]]:
    """Returns [(timestamp_seconds, frame_image_path), ...] at scene-change boundaries."""
    import cv2
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=config.SCENE_DETECT_THRESHOLD, min_scene_len=config.MIN_SCENE_LEN_SECONDS)
    )
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    if not scenes:
        scenes = [(video.base_timecode, video.duration)]

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    results = []
    for i, (start, _end) in enumerate(scenes):
        ts = start.get_seconds()
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = frames_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        results.append((ts, str(frame_path)))
    cap.release()
    return results


def _batch_extract_and_persist(
    units: list[EvidenceUnit], *, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore,
    image_paths: dict[str, str] | None = None,
) -> None:
    image_paths = image_paths or {}
    provider = get_llm_provider()
    for start in range(0, len(units), BATCH_SIZE):
        batch = units[start : start + BATCH_SIZE]
        results = provider.extract_entities_relationships_batch([u.content for u in batch])
        for unit, (entities, relationships) in zip(batch, results):
            resolve_and_assign_entities(unit, entities, relationships, graph=graph)
            enrich_and_persist(
                unit, sqlite=sqlite, vectors=vectors, graph=graph, extract_entities=False,
                image_path=image_paths.get(unit.id),
            )


def ingest_video(
    video_path: str, *, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore
) -> list[EvidenceUnit]:
    video_path = str(video_path)
    source_file = Path(video_path).name
    out_dir = config.DATA_PROCESSED_DIR / Path(video_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting audio from %s", video_path)
    audio_path = _extract_audio(video_path, out_dir)

    logger.info("Transcribing %s", audio_path)
    segments = transcribe(audio_path)

    transcript_units: list[EvidenceUnit] = []
    for seg in segments:
        transcript_units.append(EvidenceUnit(
            modality=Modality.TRANSCRIPT_SEGMENT,
            content=seg["text"],
            source_file=source_file,
            source_type=SourceType.VIDEO,
            start_time=seg["start"],
            end_time=seg["end"],
            confidence=seg["confidence"],
            provenance=Provenance(extractor="faster-whisper", model=config.WHISPER_MODEL_SIZE, raw_ref=audio_path),
        ))

    logger.info("Detecting keyframes in %s", video_path)
    keyframes = _detect_keyframes(video_path, out_dir / "frames")
    logger.info("Found %d keyframes", len(keyframes))

    frame_units: list[EvidenceUnit] = []
    frame_image_paths: dict[str, str] = {}
    ocr_units: list[EvidenceUnit] = []
    for ts, frame_path in keyframes:
        nearby_text = " ".join(
            u.content for u in transcript_units if u.start_time is not None and abs(u.start_time - ts) <= 20
        )
        caption, cap_conf = caption_image(frame_path, context=nearby_text)
        frame_unit = EvidenceUnit(
            modality=Modality.VIDEO_FRAME,
            content=caption,
            source_file=source_file,
            source_type=SourceType.VIDEO,
            start_time=ts,
            end_time=ts,
            confidence=cap_conf,
            provenance=Provenance(extractor="llm-vision-caption", model=config.LLM_PROVIDER, raw_ref=frame_path),
        )
        frame_units.append(frame_unit)
        frame_image_paths[frame_unit.id] = frame_path

        for region in run_ocr_regions(frame_path):
            if not is_substantive_ocr(region["text"]):
                continue
            ocr_units.append(EvidenceUnit(
                modality=Modality.OCR_TEXT,
                content=region["text"],
                source_file=source_file,
                source_type=SourceType.VIDEO,
                start_time=ts,
                end_time=ts,
                bbox=BoundingBox(**region["bbox"]),
                confidence=region["confidence"],
                provenance=Provenance(extractor="tesseract", raw_ref=frame_path),
            ))

    all_units = transcript_units + ocr_units + frame_units
    _batch_extract_and_persist(all_units, sqlite=sqlite, vectors=vectors, graph=graph, image_paths=frame_image_paths)

    logger.info("Ingested %d evidence units from %s", len(all_units), source_file)
    return all_units
