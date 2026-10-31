import asyncio
import os
import json
from dataclasses import dataclass
from openai import OpenAI
from app.concurrency import call_with_backoff
from app.linking.event_assembly import EVENT_TYPES


@dataclass
class EventSummary:
    event_type: str
    title: str
    summary: str


_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "event_summary", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "event_type": {"type": "string", "enum": sorted(EVENT_TYPES)},
                "title": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["event_type", "title", "summary"],
            "additionalProperties": False,
        },
    },
}


def _call_summary_model(evidence_texts: list[str]):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    joined = "\n---\n".join(evidence_texts)
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            "Summarize the following connected pieces of evidence as one knowledge event. "
            "Pick the single best event_type.\n\n" + joined}],
        response_format=_SCHEMA,
    )


async def summarize_event(evidence_texts: list[str]) -> EventSummary:
    completion = await call_with_backoff(_call_summary_model, evidence_texts)
    parsed = json.loads(completion.choices[0].message.content)
    return EventSummary(**parsed)
