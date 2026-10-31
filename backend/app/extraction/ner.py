import asyncio
import os
import json
from dataclasses import dataclass
from openai import OpenAI
from app.concurrency import call_with_backoff


@dataclass(eq=True)
class EntityMention:
    text: str
    entity_type: str


_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entities",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "entity_type": {"type": "string"},
                        },
                        "required": ["text", "entity_type"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}


def _call_ner_model(text: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            "Extract named entities (technologies, people, organizations, concepts, products) "
            "from this text. Use short lowercase entity_type categories.\n\n" + text}],
        response_format=_SCHEMA,
    )


async def extract_entities(text: str) -> list[EntityMention]:
    completion = await call_with_backoff(_call_ner_model, text)
    parsed = json.loads(completion.choices[0].message.content)
    return [EntityMention(e["text"], e["entity_type"]) for e in parsed["entities"]]
