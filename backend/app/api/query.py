from fastapi import APIRouter
from pydantic import BaseModel
from app.storage.db import get_session
from app.storage.graph import get_driver
from app.retrieval.pipeline import run_query
from app.api.history import record_query, QueryLogEntry

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


@router.post("/query")
async def query(request: QueryRequest):
    driver = get_driver()
    async with get_session() as session:
        result = await run_query(session, driver, request.query, request.top_k)
    await driver.close()
    modality_count = len({c.modality for c in result.citations})
    record_query(QueryLogEntry(
        query=request.query, mode="query", citation_count=len(result.citations),
        modality_count=modality_count, baseline_citation_count=0,
    ))
    return {
        "answer": result.answer,
        "citations": [c.__dict__ for c in result.citations],
        "events_used": result.events_used,
    }
