import pytest
from unittest.mock import patch, AsyncMock
from app.ingestion.orchestrator import JOBS, JobStatus, run_ingestion_job


@pytest.mark.asyncio
async def test_run_ingestion_job_progresses_through_real_stages():
    job_id = "test-job-stages"
    JOBS[job_id] = JobStatus(job_id=job_id, status="pending")
    seen_stages = []

    async def fake_link_source(session, driver, source_id, on_stage=None):
        seen_stages.append(JOBS[job_id].stage)  # "extract" should already have happened
        await on_stage("understand")
        seen_stages.append(JOBS[job_id].stage)
        await on_stage("link")
        seen_stages.append(JOBS[job_id].stage)
        await on_stage("index")
        return {"windows": 0, "events_created": 0, "relationships_created": 0, "contradictions_found": 0}

    with (
        patch("app.ingestion.orchestrator._validate_source_type"),
        patch("app.ingestion.orchestrator._DISPATCH", {"image": AsyncMock()}),
        patch("app.linking.orchestrator.link_source", new=fake_link_source),
        patch("app.storage.graph.get_driver") as mock_driver,
        patch("app.baseline.naive_chunk_rag.ingest_baseline_from_source", new=AsyncMock()),
    ):
        mock_driver.return_value.close = AsyncMock()
        await run_ingestion_job(job_id, "diagram.png", "image", "/tmp")

    assert seen_stages == ["extract", "understand", "link"]
    assert JOBS[job_id].stage == "done"
    assert JOBS[job_id].status == "done"

    # Real (not estimated) per-stage wall-clock timing, so a slow job is
    # diagnosable instead of just "processing" for an unexplained duration.
    timings = JOBS[job_id].stage_timings_ms
    for stage in ("extract", "understand", "link", "index", "total"):
        assert stage in timings, f"missing timing for stage {stage!r}: {timings}"
        assert timings[stage] >= 0
