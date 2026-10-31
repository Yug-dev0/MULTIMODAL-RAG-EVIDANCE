"""
Provider-agnostic interface for the LLM-dependent capabilities used across
the pipeline: diagram/frame captioning, open-vocabulary entity + relationship
extraction, cross-modal link semantic labeling, event summarization, and
final answer synthesis.

Two implementations are provided: a local one (Ollama, default — no external
dependency, safe for an offline demo) and a hosted one (Gemini, opt-in via
config.LLM_PROVIDER="gemini" once a valid AI Studio key is set). Callers
never talk to either backend directly — they go through get_llm_provider().

Each concrete provider only implements two primitives (_chat_json,
_chat_text); the actual task-specific prompts/parsing/fallback behavior live
once, in the base class, so both providers get identical semantics.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod

from pydantic import BaseModel

import config
from src.schema.evidence_unit import Entity, Relationship

logger = logging.getLogger(__name__)

LINK_LABELS = ("explains", "depicts", "supports", "contradicts", "follows", "references")


class ExtractionResult(BaseModel):
    entities: list[Entity] = []
    relationships: list[Relationship] = []


class BatchExtractionResult(BaseModel):
    items: list[ExtractionResult] = []


class CaptionResult(BaseModel):
    caption: str
    confidence: float


class RelationshipClassification(BaseModel):
    related: bool
    label: Optional[str] = None
    confidence: float


class LinkClassification(BaseModel):
    label: str
    confidence: float


class EventSummary(BaseModel):
    title: str
    summary: str


CAPTION_PROMPT = (
    "You are analyzing a still frame or image from a technical presentation, "
    "meeting recording, or document. Describe any architecture diagram, chart, "
    "code, or on-screen content in concrete, searchable technical terms: name "
    "the components/boxes and how they connect. If it's just a person talking "
    "with no diagram or on-screen text, say that briefly instead.\n"
    "Context (nearby speech/text, if any): {context}\n\n"
    'Respond ONLY with JSON: {{"caption": "...", "confidence": 0.0-1.0}}'
)

EXTRACTION_PROMPT = (
    "Extract entities and relationships from the text below, which comes from "
    "a multimodal technical source (a transcript segment, slide, document, or "
    "diagram OCR/caption). Entities may be any type relevant to the content "
    "(Person, System, Component, Technology, Concept, Organization, Metric, "
    "etc.) — do not restrict yourself to a fixed list, and do not invent "
    "entities that aren't grounded in the text. Relationships are (subject, "
    "predicate, object) triples grounded in the text. Assign a confidence "
    "0.0-1.0 to each item. If nothing meaningful is present, return empty "
    "lists.\n\nTEXT:\n{text}\n\n"
    'Respond ONLY with JSON: {{"entities": [{{"text": "...", "type": "...", '
    '"confidence": 0.0}}], "relationships": [{{"subject": "...", "predicate": '
    '"...", "object": "...", "confidence": 0.0}}]}}'
)

BATCH_EXTRACTION_PROMPT = (
    "Below are {n} separately-numbered short texts from a multimodal technical source "
    "(transcript segments, OCR text, captions, document text). For EACH numbered item "
    "independently, extract entities and relationships exactly as you would one at a time — "
    "open-vocabulary entity types, grounded only in that item's own text, empty lists if nothing "
    "meaningful is present. Do not let one item's content leak into another's results.\n\n"
    "{items}\n\n"
    'Respond ONLY with a JSON object: {{"items": [ {{"entities": [{{"text":"...","type":"...",'
    '"confidence":0.0}}], "relationships": [{{"subject":"...","predicate":"...","object":"...",'
    '"confidence":0.0}}]}}, ... ]}} — the "items" array MUST have exactly {n} objects, in the same '
    "order as the numbered texts."
)

LINK_CLASSIFY_PROMPT = (
    "Two pieces of evidence from a multimodal source co-occur or are "
    "semantically similar. Classify HOW the first relates to the second, "
    "choosing exactly one label from: " + ", ".join(LINK_LABELS) + ".\n\n"
    "EVIDENCE A ({modality_a}): {content_a}\n\nEVIDENCE B ({modality_b}): {content_b}\n\n"
    'Respond ONLY with JSON: {{"label": "...", "confidence": 0.0-1.0}}'
)

EVENT_SUMMARY_PROMPT = (
    "The following evidence snippets were extracted from the same short span "
    "of a multimodal recording/document and are strongly linked. Write a "
    "short title (< 10 words) and a 1-2 sentence summary of what is being "
    "discussed/shown, grounded only in the snippets below — do not invent "
    "details.\n\nSNIPPETS:\n{snippets}\n\n"
    'Respond ONLY with JSON: {{"title": "...", "summary": "..."}}'
)

SYNTHESIS_PROMPT = (
    "Answer the user's question using ONLY the evidence bundles below. Each "
    "bundle may combine speech, visual, and document evidence with its "
    "source/timestamp/page. Cite facts inline using the bracketed [E#] tag "
    "from the bundle you drew them from. If the evidence doesn't fully answer "
    "the question, say plainly what's missing rather than guessing.\n\n"
    "QUESTION: {query}\n\nEVIDENCE:\n{evidence}\n\nANSWER:"
)


def _format_evidence(evidence_bundles: list[dict]) -> str:
    lines = []
    for i, b in enumerate(evidence_bundles):
        lines.append(
            f"[E{i}] modality={b.get('modality')} source={b.get('source_file')} "
            f"time={b.get('start_time')}-{b.get('end_time')} page={b.get('page_number')} "
            f"speaker={b.get('speaker')}\ncontent: {b.get('content')}"
        )
    return "\n\n".join(lines)


def _extract_json_object(raw: str) -> dict:
    """Best-effort JSON extraction — models sometimes wrap JSON in prose/fences."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start : end + 1])
    raise ValueError(f"no JSON object found in: {raw[:200]!r}")


class LLMProvider(ABC):
    @abstractmethod
    def _chat_json(self, prompt: str, image_path: str | None = None) -> dict: ...

    @abstractmethod
    def _chat_text(self, prompt: str) -> str: ...

    def caption_image(self, image_path: str, context: str = "") -> tuple[str, float]:
        prompt = CAPTION_PROMPT.format(context=context or "none")
        try:
            result = CaptionResult.model_validate(self._chat_json(prompt, image_path=image_path))
            return result.caption, result.confidence
        except Exception as e:
            logger.warning("caption_image failed: %s", e)
            return "[caption unavailable]", 0.0

    def extract_entities_relationships(self, text: str) -> tuple[list[Entity], list[Relationship]]:
        prompt = EXTRACTION_PROMPT.format(text=text[:4000])
        try:
            result = ExtractionResult.model_validate(self._chat_json(prompt))
            return result.entities, result.relationships
        except Exception as e:
            logger.warning("extract_entities_relationships failed: %s", e)
            return [], []

    def extract_entities_relationships_batch(
        self, texts: list[str]
    ) -> list[tuple[list[Entity], list[Relationship]]]:
        """
        One LLM call for many short texts at once (the free-tier quota is
        request-rate-limited, not just per-call-size-limited, so batching many
        small units into one call is the difference between a corpus taking
        minutes vs. hours to enrich). Falls back to empty results for the
        whole batch on any failure — callers should keep batches modest
        (~15-20 items) so a single bad response doesn't lose too much.
        """
        if not texts:
            return []
        items_block = "\n".join(f"[{i}] {t[:600]}" for i, t in enumerate(texts))
        prompt = BATCH_EXTRACTION_PROMPT.format(n=len(texts), items=items_block)
        try:
            result = BatchExtractionResult.model_validate(self._chat_json(prompt))
            out = [(item.entities, item.relationships) for item in result.items]
        except Exception as e:
            logger.warning("extract_entities_relationships_batch failed: %s", e)
            out = []
        if len(out) < len(texts):
            out += [([], [])] * (len(texts) - len(out))
        return out[: len(texts)]

    def classify_link(self, modality_a: str, content_a: str, modality_b: str, content_b: str) -> tuple[str | None, float]:
        prompt = LINK_CLASSIFY_PROMPT.format(
            modality_a=modality_a, content_a=content_a[:800], modality_b=modality_b, content_b=content_b[:800]
        )
        try:
            result = LinkClassification.model_validate(self._chat_json(prompt))
            if result.label not in LINK_LABELS:
                return None, 0.0
            return result.label, result.confidence
        except Exception as e:
            logger.warning("classify_link failed: %s", e)
            return None, 0.0

    def summarize_event(self, evidence_snippets: list[str]) -> tuple[str, str]:
        prompt = EVENT_SUMMARY_PROMPT.format(snippets="\n".join(f"- {s[:300]}" for s in evidence_snippets))
        try:
            result = EventSummary.model_validate(self._chat_json(prompt))
            return result.title, result.summary
        except Exception as e:
            logger.warning("summarize_event failed: %s", e)
            fallback = evidence_snippets[0][:80] if evidence_snippets else "Untitled event"
            return fallback, " ".join(evidence_snippets)[:300]

    def synthesize_answer(self, query: str, evidence_bundles: list[dict]) -> str:
        prompt = SYNTHESIS_PROMPT.format(query=query, evidence=_format_evidence(evidence_bundles))
        try:
            return self._chat_text(prompt)
        except Exception as e:
            logger.warning("synthesize_answer failed: %s", e)
            return f"[answer synthesis unavailable: {e}]"

    def answer_from_context(self, query: str, context: str) -> str:
        """Plain context->answer completion, deliberately with no citation format
        or provenance — used only by the naive text-only baseline for a fair
        comparison against the same underlying model."""
        prompt = (
            "Answer the question using ONLY the context below. If the context "
            "doesn't contain the answer, say so plainly.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
        )
        try:
            return self._chat_text(prompt)
        except Exception as e:
            logger.warning("answer_from_context failed: %s", e)
            return f"[answer unavailable: {e}]"


RETRYABLE_MARKERS = ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503")
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s|retry in (\d+(?:\.\d+)?)s")


def _suggested_delay(error_text: str, default: float) -> float:
    match = _RETRY_DELAY_RE.search(error_text)
    if not match:
        return default
    value = match.group(1) or match.group(2)
    return min(float(value) + 1.0, 60.0)  # +1s safety margin, capped so one call can't stall a batch too long


_last_call_time = 0.0
MIN_CALL_INTERVAL_SECONDS = 13.0  # free-tier key observed at 5 req/min; paced at ~4.6/min for margin


def _pace() -> None:
    """Proactively spaces out calls so we rarely hit 429 in the first place —
    reactive backoff alone was too slow against a hard per-minute cap."""
    global _last_call_time
    now = time.monotonic()
    wait = MIN_CALL_INTERVAL_SECONDS - (now - _last_call_time)
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.monotonic()


def _with_retry(fn, *, attempts: int = 3, base_delay: float = 3.0):
    last_error = None
    for attempt in range(attempts):
        _pace()
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - broad on purpose, re-raised after final attempt
            last_error = e
            if attempt == attempts - 1 or not any(m in str(e) for m in RETRYABLE_MARKERS):
                raise
            delay = _suggested_delay(str(e), base_delay * (2 ** attempt))
            logger.info("Retryable LLM error (attempt %d/%d), backing off %.1fs", attempt + 1, attempts, delay)
            time.sleep(delay)
    raise last_error  # pragma: no cover - unreachable, satisfies type checkers


# Free-tier keys observed with separate, independent daily-quota buckets per
# model name. A per-minute 429 is worth backing off and retrying the same
# model; a per-day 429 is not — this project instead fails over to the next
# candidate model, since each gets its own fresh bucket.
GEMINI_MODEL_CANDIDATES = [
    "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash",
    "gemini-3-flash-preview", "gemini-flash-latest", "gemini-2.5-flash",
]


def _is_daily_quota_error(error_text: str) -> bool:
    return "RequestsPerDay" in error_text


class GeminiProvider(LLMProvider):
    def __init__(self):
        from google import genai

        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        preferred = config.GEMINI_MODEL
        self._candidates = [preferred] + [m for m in GEMINI_MODEL_CANDIDATES if m != preferred]
        self._model_index = 0

    @property
    def model(self) -> str:
        return self._candidates[self._model_index]

    def _advance_model(self) -> bool:
        if self._model_index >= len(self._candidates) - 1:
            return False
        self._model_index += 1
        logger.warning(
            "Gemini model %s hit its daily quota; switching to %s",
            self._candidates[self._model_index - 1], self.model,
        )
        return True

    def _call_with_model_failover(self, do_call):
        while True:
            try:
                return _with_retry(lambda: do_call(self.model))
            except Exception as e:
                if _is_daily_quota_error(str(e)) and self._advance_model():
                    continue
                raise

    def _chat_json(self, prompt, image_path=None):
        from google.genai import types

        contents = [prompt]
        if image_path:
            from PIL import Image

            contents = [Image.open(image_path), prompt]

        def do_call(model):
            resp = self.client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return _extract_json_object(resp.text)

        return self._call_with_model_failover(do_call)

    def _chat_text(self, prompt):
        def do_call(model):
            resp = self.client.models.generate_content(model=model, contents=prompt)
            return resp.text

        return self._call_with_model_failover(do_call)


class OllamaProvider(LLMProvider):
    def __init__(self):
        import ollama

        self._client = ollama.Client(host=config.OLLAMA_HOST)
        self.vision_model = config.OLLAMA_VISION_MODEL
        self.text_model = config.OLLAMA_TEXT_MODEL

    def _chat_json(self, prompt, image_path=None):
        message = {"role": "user", "content": prompt}
        model = self.text_model
        if image_path:
            message["images"] = [image_path]
            model = self.vision_model
        resp = self._client.chat(model=model, messages=[message], format="json")
        return _extract_json_object(resp["message"]["content"])

    def _chat_text(self, prompt):
        resp = self._client.chat(model=self.text_model, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]


_provider_singleton: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider_singleton
    if _provider_singleton is None:
        if config.LLM_PROVIDER == "gemini":
            _provider_singleton = GeminiProvider()
        else:
            _provider_singleton = OllamaProvider()
    return _provider_singleton
