import asyncio
import base64
import os
import json
from dataclasses import dataclass
from openai import OpenAI
from app.concurrency import call_with_backoff


@dataclass
class ImageCaption:
    caption: str
    description: str
    confidence: float


_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_caption",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
                "description": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["caption", "description", "confidence"],
            "additionalProperties": False,
        },
    },
}


def _call_vision_model(image_path: str, context: str | None):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    prompt = "Describe this image in detail, including any diagrams, architecture, or on-screen text meaning."
    if context:
        prompt += f" Context from surrounding discussion: {context}"
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        response_format=_SCHEMA,
    )


async def caption_image(image_path: str, context: str | None = None) -> ImageCaption:
    """
    Runs the blocking OpenAI SDK call in a worker thread rather than directly
    on the event loop, so `await`-ing this alongside other in-flight calls
    (via asyncio.gather) actually overlaps their network latency instead of
    serializing it — the previous plain `def` blocked the whole loop for the
    duration of each request.
    """
    completion = await call_with_backoff(_call_vision_model, image_path, context)
    parsed = json.loads(completion.choices[0].message.content)
    return ImageCaption(**parsed)
