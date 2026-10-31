import asyncio
import os
import json
from dataclasses import dataclass
from openai import OpenAI
from app.concurrency import call_with_backoff
from app.storage.repository import persist_relationship


@dataclass
class ContradictionResult:
    contradicts: bool
    confidence: float
    explanation: str


_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "contradiction_check", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "contradicts": {"type": "boolean"},
                "confidence": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": ["contradicts", "confidence", "explanation"],
            "additionalProperties": False,
        },
    },
}


def _call_contradiction_model(earlier_summary: str, later_summary: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        f"Earlier statement: {earlier_summary}\nLater statement: {later_summary}\n\n"
        "Does the later statement contradict or reverse the earlier one about the same subject?"
    )
    return client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format=_SCHEMA,
    )


async def check_contradiction(earlier_summary: str, later_summary: str) -> ContradictionResult:
    completion = await call_with_backoff(_call_contradiction_model, earlier_summary, later_summary)
    parsed = json.loads(completion.choices[0].message.content)
    return ContradictionResult(**parsed)


async def detect_and_record_contradictions(session, events_for_entity: list) -> list:
    # timestamp_start is None for events from non-timestamped sources (images, PDFs);
    # fall back to creation order (always set) so the sort key stays comparable.
    ordered = sorted(
        events_for_entity,
        key=lambda e: (e.timestamp_start if e.timestamp_start is not None else float("-inf"), e.created_at),
    )
    pairs = list(zip(ordered, ordered[1:]))
    results = await asyncio.gather(*(check_contradiction(earlier.summary, later.summary) for earlier, later in pairs))
    created = []
    for (earlier, later), result in zip(pairs, results):
        if result.contradicts and result.confidence >= 0.6:
            rel = await persist_relationship(
                session, source_type="knowledge_event", source_id=earlier.id,
                target_type="knowledge_event", target_id=later.id, relationship_type="contradicts",
                confidence=result.confidence, timestamp_start=earlier.timestamp_start,
                timestamp_end=later.timestamp_end, evidence_ids=[],
                metadata_json={"superseded_by": str(later.id), "explanation": result.explanation},
            )
            created.append(rel)
    return created
