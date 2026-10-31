import pytest
from app.ingestion.orchestrator import (
    JOBS, JobStatus, run_ingestion_job, _validate_source_type, SourceTypeMismatchError,
)


def test_validate_source_type_raises_on_mismatch():
    with pytest.raises(SourceTypeMismatchError, match="looks like a 'pdf' file"):
        _validate_source_type("resume.pdf", "video")


def test_validate_source_type_passes_on_match():
    _validate_source_type("meeting.mp4", "video")  # no raise


@pytest.mark.asyncio
async def test_run_ingestion_job_reports_clear_error_on_mismatch():
    job_id = "test-job-mismatch"
    JOBS[job_id] = JobStatus(job_id=job_id, status="pending")
    await run_ingestion_job(job_id, "resume.pdf", "video", "/tmp")
    assert JOBS[job_id].status == "failed"
    assert "you selected 'video'" in JOBS[job_id].error.lower()
    assert "pdf" in JOBS[job_id].error.lower()
