import json
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from app.api.auth import get_current_user, User
from app.storage.db import get_session
from app.storage.graph import get_driver
from app.storage.repository import (
    create_chat_session, get_chat_session, list_chat_sessions, get_chat_messages,
    get_session_source_ids, persist_chat_message, touch_chat_session, _utc_iso,
)
from app.retrieval.pipeline import run_query
from app.concurrency import call_with_backoff

router = APIRouter()

_TITLE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "session_title", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}


def _call_title_model(query: str, answer: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            "Write a short chat title (max 6 words, no quotes) that captures the topic of this "
            f"exchange.\n\nQuestion: {query}\nAnswer: {answer[:400]}"}],
        response_format=_TITLE_SCHEMA,
    )


async def _generate_session_title(query: str, answer: str) -> str:
    completion = await call_with_backoff(_call_title_model, query, answer)
    return json.loads(completion.choices[0].message.content)["title"]


class ChatQueryRequest(BaseModel):
    query: str
    top_k: int = 10


def _session_dict(chat_session, message_count: int = 0, source_count: int = 0) -> dict:
    return {
        "id": str(chat_session.id), "title": chat_session.title,
        "created_at": _utc_iso(chat_session.created_at), "updated_at": _utc_iso(chat_session.updated_at),
        "message_count": message_count, "source_count": source_count,
    }


@router.post("/sessions")
async def new_session(user: User = Depends(get_current_user)):
    async with get_session() as db:
        chat_session = await create_chat_session(db, owner_uid=user.uid)
        await db.commit()
        return _session_dict(chat_session)


@router.get("/sessions")
async def sessions(user: User = Depends(get_current_user)):
    async with get_session() as db:
        return await list_chat_sessions(db, owner_uid=user.uid)


@router.get("/sessions/{session_id}/messages")
async def messages(session_id: str, user: User = Depends(get_current_user)):
    async with get_session() as db:
        chat_session = await get_chat_session(db, uuid.UUID(session_id), user.uid)
        if chat_session is None:
            raise HTTPException(status_code=404, detail="chat session not found")
        msgs = await get_chat_messages(db, chat_session.id)
        return [
            {
                "id": str(m.id), "role": m.role, "content": m.content,
                "citations": m.citations_json, "created_at": _utc_iso(m.created_at),
            }
            for m in msgs
        ]


@router.post("/sessions/{session_id}/query")
async def session_query(session_id: str, request: ChatQueryRequest, user: User = Depends(get_current_user)):
    driver = get_driver()
    async with get_session() as db:
        chat_session = await get_chat_session(db, uuid.UUID(session_id), user.uid)
        if chat_session is None:
            raise HTTPException(status_code=404, detail="chat session not found")

        source_ids = await get_session_source_ids(db, chat_session.id)
        result = await run_query(db, driver, request.query, request.top_k, allowed_source_ids=source_ids or [])

        await persist_chat_message(db, session_id=chat_session.id, role="user", content=request.query)
        await persist_chat_message(
            db, session_id=chat_session.id, role="assistant", content=result.answer,
            citations_json=[c.__dict__ for c in result.citations],
        )

        new_title = None
        if chat_session.title == "New chat":
            new_title = await _generate_session_title(request.query, result.answer)
        await touch_chat_session(db, chat_session.id, title=new_title)
        await db.commit()
    await driver.close()

    return {
        "answer": result.answer,
        "citations": [c.__dict__ for c in result.citations],
        "events_used": result.events_used,
        "session_title": new_title or chat_session.title,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: User = Depends(get_current_user)):
    from sqlalchemy import delete
    from app.storage.models import ChatMessage, SessionSource
    async with get_session() as db:
        sess_uuid = uuid.UUID(session_id)
        chat_session = await get_chat_session(db, sess_uuid, user.uid)
        if chat_session is None:
            raise HTTPException(status_code=404, detail="chat session not found")
        await db.execute(delete(ChatMessage).where(ChatMessage.session_id == sess_uuid))
        await db.execute(delete(SessionSource).where(SessionSource.session_id == sess_uuid))
        await db.delete(chat_session)
        await db.commit()
    return {"status": "ok"}
