import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional
from app.storage.db import get_session
from app.storage.repository import link_source_to_session
from app.ingestion.video import ingest_video
from app.ingestion.audio import ingest_audio
from app.ingestion.image import ingest_image
from app.ingestion.pdf import ingest_pdf

logger = logging.getLogger(__name__)

Stage = Literal["upload", "extract", "understand", "link", "index", "done"]


@dataclass
class JobStatus:
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    source_id: Optional[str] = None
    error: Optional[str] = None
    filename: Optional[str] = None
    source_type: Optional[str] = None
    session_id: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    stage: Stage = "upload"
    stage_timings_ms: dict = field(default_factory=dict)


class _StageTimer:
    """Real (not estimated) per-stage wall-clock timing, driven by the same
    stage transitions already reported to the frontend -- no separate
    instrumentation system, just recording when each one starts and ends."""

    def __init__(self, first_stage: str):
        self._t0 = time.time()
        self._stage = first_stage
        self.timings_ms: dict[str, float] = {}

    def transition(self, next_stage: str) -> None:
        now = time.time()
        self.timings_ms[self._stage] = round((now - self._t0) * 1000, 1)
        self._t0 = now
        self._stage = next_stage

    def finish(self) -> None:
        self.timings_ms[self._stage] = round((time.time() - self._t0) * 1000, 1)


JOBS: dict[str, JobStatus] = {}

_DISPATCH = {"video": ingest_video, "audio": ingest_audio, "image": ingest_image, "pdf": ingest_pdf}

_EXPECTED_EXTENSIONS = {
    "video": {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"},
    "audio": {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"},
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"},
    "pdf": {".pdf"},
}


class SourceTypeMismatchError(ValueError):
    pass


def _validate_source_type(file_path: str, source_type: str) -> None:
    ext = os.path.splitext(file_path)[1].lower()
    expected = _EXPECTED_EXTENSIONS.get(source_type, set())
    if ext not in expected:
        detected = next((t for t, exts in _EXPECTED_EXTENSIONS.items() if ext in exts), None)
        hint = f" This looks like a '{detected}' file." if detected else ""
        raise SourceTypeMismatchError(
            f"You selected '{source_type}' but the file has a '{ext or '(no extension)'}' extension, "
            f"which {source_type} ingestion can't process.{hint} Pick the matching source type and try again."
        )


async def run_ingestion_job(job_id: str, file_path: str, source_type: str, media_dir: str,
                             session_id: Optional[uuid.UUID] = None) -> None:
    JOBS[job_id].status = "running"
    JOBS[job_id].stage = "extract"
    source_id = uuid.uuid4()
    JOBS[job_id].source_id = str(source_id)
    timer = _StageTimer("extract")
    try:
        _validate_source_type(file_path, source_type)
        ingest_fn = _DISPATCH[source_type]
        async with get_session() as session:
            if session_id is not None:
                await link_source_to_session(session, source_id=source_id, session_id=session_id)
            if source_type in ("video", "pdf"):
                await ingest_fn(session, file_path, source_id, media_dir)
            else:
                await ingest_fn(session, file_path, source_id)
            await session.commit()

        from app.linking.orchestrator import link_source
        from app.storage.graph import get_driver

        async def on_stage(stage: Stage) -> None:
            timer.transition(stage)
            JOBS[job_id].stage = stage
            JOBS[job_id].stage_timings_ms = dict(timer.timings_ms)

        driver = get_driver()
        async with get_session() as link_session:
            await link_source(link_session, driver, source_id, on_stage=on_stage)
            await link_session.commit()
        await driver.close()

        JOBS[job_id].stage = "index"
        from app.baseline.naive_chunk_rag import ingest_baseline_from_source
        async with get_session() as baseline_session:
            await ingest_baseline_from_source(baseline_session, source_id)
            await baseline_session.commit()

        timer.finish()
        timer.timings_ms["total"] = round((time.time() - JOBS[job_id].submitted_at) * 1000, 1)
        JOBS[job_id].stage_timings_ms = dict(timer.timings_ms)
        JOBS[job_id].stage = "done"
        JOBS[job_id].status = "done"
    except SourceTypeMismatchError as exc:
        logger.warning("Ingestion job %s rejected: %s", job_id, exc)
        JOBS[job_id].status = "failed"
        JOBS[job_id].error = str(exc)
    except Exception as exc:
        logger.exception("Ingestion job %s failed", job_id)
        timer.finish()
        JOBS[job_id].stage_timings_ms = dict(timer.timings_ms)
        JOBS[job_id].status = "failed"
        JOBS[job_id].error = str(exc)
