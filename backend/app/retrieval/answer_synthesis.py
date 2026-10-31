import asyncio
import os
import json
from dataclasses import dataclass
from urllib.parse import quote
from openai import OpenAI
from app.concurrency import call_with_backoff

# Modalities whose raw_locator points at an actual image file on disk (a video
# keyframe or a PDF's embedded image), as opposed to a time/page fragment of a
# non-image source — only these can produce a viewable thumbnail.
_VISUAL_MODALITIES = {"ocr", "image"}


@dataclass
class Citation:
    evidence_id: str
    modality: str
    source: str
    content: str
    timestamp_start: float | None
    timestamp_end: float | None
    page_number: int | None
    thumbnail: str | None
    confidence: float


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    events_used: list[str]


_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_answer", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "cited_evidence_ids"],
            "additionalProperties": False,
        },
    },
}


def _call_answer_model(query: str, ranked_evidence: list, ranked_events: list):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    evidence_block = "\n".join(
        f"[{u.id}] ({u.modality}, t={u.timestamp_start}) {u.content_text}" for u in ranked_evidence
    )
    events_block = "\n".join(f"- {e.title}: {e.summary}" for e in ranked_events)
    prompt = (
        f"Question: {query}\n\nRelevant knowledge events:\n{events_block}\n\n"
        f"Evidence (cite by the bracketed id, only ids that genuinely support the answer):\n{evidence_block}"
    )
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise question-answering assistant. You must answer the user's question strictly using the provided relevant knowledge events and evidence. If the provided evidence does not contain the answer or is not relevant to the question, you MUST reply: 'I cannot find the answer in the uploaded files.' and nothing else. Under no circumstances should you use any outside knowledge, pre-trained facts, assumptions, or database records from other topics not present in the provided evidence. For example, do not answer questions about mythology, general history, or other entities unless they are explicitly and directly mentioned in the provided context."},
            {"role": "user", "content": prompt}
        ],
        response_format=_SCHEMA,
    )


async def synthesize_answer(query: str, ranked_evidence: list, ranked_events: list) -> AnswerResult:
    if not ranked_evidence and not ranked_events:
        return AnswerResult(
            answer="I cannot find the answer in the uploaded files.",
            citations=[],
            events_used=[],
        )

    completion = await call_with_backoff(_call_answer_model, query, ranked_evidence, ranked_events)
    parsed = json.loads(completion.choices[0].message.content)

    by_id = {str(u.id): u for u in ranked_evidence}
    citations = []
    for evidence_id in parsed["cited_evidence_ids"]:
        unit = by_id.get(evidence_id)
        if unit is None:
            continue
        thumbnail = f"/assets?path={quote(unit.raw_locator)}" if unit.modality in _VISUAL_MODALITIES else None
        citations.append(Citation(
            evidence_id=evidence_id, modality=unit.modality, source=unit.raw_locator,
            content=unit.content_text,
            timestamp_start=unit.timestamp_start, timestamp_end=unit.timestamp_end,
            page_number=unit.page_number, thumbnail=thumbnail, confidence=unit.confidence,
        ))
    return AnswerResult(
        answer=parsed["answer"], citations=citations,
        events_used=[str(e.id) for e in ranked_events],
    )
