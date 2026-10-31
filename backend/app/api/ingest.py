import os
import uuid
from fastapi import APIRouter, UploadFile, Form, BackgroundTasks, HTTPException, Depends
from app.ingestion.orchestrator import JOBS, JobStatus, run_ingestion_job
from app.storage.db import get_session
from app.storage.repository import create_chat_session, get_chat_session
from app.api.auth import get_current_user, User

router = APIRouter()
MEDIA_DIR = os.environ.get("MEDIA_STORAGE_DIR", "./data/media")


@router.post("/ingest")
async def ingest(file: UploadFile, source_type: str = Form(...), session_id: str | None = Form(None),
                  background_tasks: BackgroundTasks = None, user: User = Depends(get_current_user)):
    async with get_session() as db:
        if session_id:
            chat_session = await get_chat_session(db, uuid.UUID(session_id), user.uid)
            if chat_session is None:
                raise HTTPException(status_code=404, detail="chat session not found")
        else:
            chat_session = await create_chat_session(db, owner_uid=user.uid)
        await db.commit()
        effective_session_id = chat_session.id

    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(
        job_id=job_id, status="pending", filename=file.filename, source_type=source_type,
        session_id=str(effective_session_id),
    )
    os.makedirs(MEDIA_DIR, exist_ok=True)
    file_path = os.path.join(MEDIA_DIR, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    background_tasks.add_task(run_ingestion_job, job_id, file_path, source_type, MEDIA_DIR, effective_session_id)
    return {"job_id": job_id, "session_id": str(effective_session_id)}


@router.get("/ingest/{job_id}")
async def ingest_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job not found")
    return JOBS[job_id]


@router.get("/jobs")
async def list_jobs():
    return sorted(JOBS.values(), key=lambda j: j.submitted_at, reverse=True)
