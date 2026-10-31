import asyncio
import os
import json
from dataclasses import dataclass
from openai import OpenAI
from app.concurrency import call_with_backoff


@dataclass
class RelationshipInference:
    related: bool
    relationship_type: str
    confidence: float
    evidence: str


_RELATIONSHIP_TYPES = ["explains", "supports", "corresponds_to", "mentions", "depicts",
                        "contradicts", "follows", "references", "unrelated"]

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "relationship_inference",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "related": {"type": "boolean"},
                "relationship_type": {"type": "string", "enum": _RELATIONSHIP_TYPES},
                "confidence": {"type": "number"},
                "evidence": {"type": "string"},
            },
            "required": ["related", "relationship_type", "confidence", "evidence"],
            "additionalProperties": False,
        },
    },
}


def _call_relationship_model(window_text: str, candidate_text: str, candidate_modality: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        f"Spoken/topic window text:\n{window_text}\n\n"
        f"Candidate {candidate_modality} evidence:\n{candidate_text}\n\n"
        "Are these related? If so, classify the relationship using exactly one of: "
        + ", ".join(_RELATIONSHIP_TYPES)
    )
    return client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format=_SCHEMA,
    )


async def infer_relationship(window_text: str, candidate_text: str, candidate_modality: str) -> RelationshipInference:
    completion = await call_with_backoff(
        _call_relationship_model, window_text, candidate_text, candidate_modality
    )
    parsed = json.loads(completion.choices[0].message.content)
    return RelationshipInference(**parsed)
