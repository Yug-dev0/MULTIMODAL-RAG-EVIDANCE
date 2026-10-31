from fastapi import APIRouter
from app.storage.db import get_session
from app.storage.graph import get_driver
from app.retrieval.pipeline import run_query
from app.baseline.naive_chunk_rag import run_baseline_query
from app.api.history import record_query, QueryLogEntry

router = APIRouter()


@router.get("/compare")
async def compare(query: str):
    driver = get_driver()
    async with get_session() as session:
        multimodal_result = await run_query(session, driver, query)
        baseline_answer = await run_baseline_query(session, query)
    await driver.close()
    modality_count = len({c.modality for c in multimodal_result.citations})
    # The baseline is a deliberately naive plain-text answer with no
    # structured citations at all -- 0 here is the actual point being
    # demonstrated, not a placeholder.
    record_query(QueryLogEntry(
        query=query, mode="compare", citation_count=len(multimodal_result.citations),
        modality_count=modality_count, baseline_citation_count=0,
    ))
    return {
        "multimodal": {
            "answer": multimodal_result.answer,
            "citations": [c.__dict__ for c in multimodal_result.citations],
            "events_used": multimodal_result.events_used,
        },
        "baseline": {"answer": baseline_answer},
    }
