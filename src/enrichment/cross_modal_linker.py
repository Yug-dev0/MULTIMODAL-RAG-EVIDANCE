"""
Cross-modal linking: after ingestion, finds which speech units and which
visual/document units are actually connected, and records that connection as
a CrossModalLink on both units (plus a graph edge) so retrieval can recover
one from the other. Runs once over the whole corpus (not per-source) so it
also catches cross-file connections — e.g. a lecture's spoken explanation
and a paper's figure for the same architecture.

Three independent, complementary passes:
  - link_temporal:   same-source speech <-> visual/text-candidate units
                      within config.TEMPORAL_COOCCURRENCE_WINDOW_SECONDS.
  - link_semantic:   Chroma-ANN nearest neighbors (text AND CLIP embedding
                      spaces) of each semantic speech window, above
                      config.SEMANTIC_LINK_MIN_SIMILARITY, confirmed by one
                      bounded LLM call per candidate — catches matches with
                      no shared timestamp at all (different files).
  - link_same_entity: units in different files/modalities that already
                      resolved to the same canonical entity — cheap,
                      deterministic, no LLM call needed.
"""

from __future__ import annotations

import logging

import config
from src.enrichment.embeddings import cosine_similarity, embed_text, embed_text_clip
from src.enrichment.llm_provider import get_llm_provider
from src.schema.evidence_unit import CrossModalLink, EvidenceUnit, Modality
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

SPEECH_MODALITIES = {Modality.TRANSCRIPT_SEGMENT}
VISUAL_MODALITIES = {Modality.VIDEO_FRAME, Modality.IMAGE, Modality.PDF_IMAGE}
TEXTUAL_CANDIDATE_MODALITIES = {Modality.OCR_TEXT, Modality.PDF_TEXT}
BOUNDARY_THRESHOLD = 0.5


def segment_speech_units(speech_units: list[EvidenceUnit], scene_timestamps: list[float]) -> list[list[EvidenceUnit]]:
    """
    Groups consecutive same-source speech units into semantic windows using a
    multi-signal boundary score (embedding-similarity drop + pause length + a
    visual scene change falling between them) instead of a fixed time window
    or sentence count — this is the "semantic events, not fixed-size chunks"
    behavior applied to speech.
    """
    ordered = sorted(speech_units, key=lambda u: u.start_time or 0.0)
    if not ordered:
        return []
    embeddings = [embed_text(u.content) for u in ordered]
    windows: list[list[EvidenceUnit]] = [[ordered[0]]]
    for i in range(1, len(ordered)):
        prev, curr = ordered[i - 1], ordered[i]
        if prev.source_file != curr.source_file:
            windows.append([curr])
            continue
        similarity_drop = max(0.0, 1.0 - cosine_similarity(embeddings[i - 1], embeddings[i]))
        gap = max(0.0, (curr.start_time or 0.0) - (prev.end_time or 0.0))
        pause_norm = min(1.0, gap / 4.0)
        scene_change = 1.0 if any((prev.end_time or 0) <= t <= (curr.start_time or 0) for t in scene_timestamps) else 0.0
        boundary_strength = 0.45 * similarity_drop + 0.30 * pause_norm + 0.25 * scene_change
        if boundary_strength >= BOUNDARY_THRESHOLD:
            windows.append([curr])
        else:
            windows[-1].append(curr)
    return windows


def _record_link(
    unit_a: EvidenceUnit, unit_b: EvidenceUnit, link_type: str, confidence: float, label: str | None,
    sqlite: SQLiteStore, graph: GraphStore,
) -> bool:
    if unit_a.id == unit_b.id:
        return False
    existing_targets = {l.target_evidence_id for l in unit_a.cross_modal_links}
    if unit_b.id in existing_targets:
        return False
    unit_a.cross_modal_links.append(
        CrossModalLink(target_evidence_id=unit_b.id, link_type=link_type, confidence=confidence, semantic_label=label)
    )
    unit_b.cross_modal_links.append(
        CrossModalLink(target_evidence_id=unit_a.id, link_type=link_type, confidence=confidence, semantic_label=label)
    )
    sqlite.upsert([unit_a, unit_b])
    if link_type == "temporal_cooccurrence":
        graph.add_co_occurs(unit_a.id, unit_b.id, confidence)
    else:
        graph.add_semantic_link(unit_a.id, unit_b.id, confidence)
    return True


def link_temporal(units: list[EvidenceUnit], *, sqlite: SQLiteStore, graph: GraphStore) -> int:
    speech = [u for u in units if u.modality in SPEECH_MODALITIES and u.start_time is not None]
    candidates = [
        u for u in units
        if u.modality in (VISUAL_MODALITIES | TEXTUAL_CANDIDATE_MODALITIES) and u.start_time is not None
    ]
    window = config.TEMPORAL_COOCCURRENCE_WINDOW_SECONDS
    by_source: dict[str, list[EvidenceUnit]] = {}
    for c in candidates:
        by_source.setdefault(c.source_file, []).append(c)

    created = 0
    for s in speech:
        for c in by_source.get(s.source_file, []):
            s_end = s.end_time if s.end_time is not None else s.start_time
            c_end = c.end_time if c.end_time is not None else c.start_time
            distance = max(0.0, max(s.start_time, c.start_time) - min(s_end, c_end))
            if distance <= window:
                confidence = max(0.3, 1.0 - distance / max(window, 1e-6))
                if _record_link(s, c, "temporal_cooccurrence", confidence, None, sqlite, graph):
                    created += 1
    return created


def link_semantic(
    units: list[EvidenceUnit], *, vectors: VectorStore, sqlite: SQLiteStore, graph: GraphStore, top_k: int = 3,
) -> int:
    speech = [u for u in units if u.modality in SPEECH_MODALITIES]
    by_id = {u.id: u for u in units}
    scene_ts_by_source: dict[str, list[float]] = {}
    for u in units:
        if u.modality in VISUAL_MODALITIES and u.start_time is not None:
            scene_ts_by_source.setdefault(u.source_file, []).append(u.start_time)

    by_source_speech: dict[str, list[EvidenceUnit]] = {}
    for s in speech:
        by_source_speech.setdefault(s.source_file, []).append(s)

    provider = get_llm_provider()
    created = 0
    for source_file, su in by_source_speech.items():
        windows = segment_speech_units(su, sorted(scene_ts_by_source.get(source_file, [])))
        for window_units in windows:
            window_text = " ".join(u.content for u in window_units if u.content.strip())
            if not window_text.strip():
                continue
            window_id_set = {u.id for u in window_units}

            text_hits = vectors.query_text(embed_text(window_text), top_k=top_k + len(window_units))
            image_hits = vectors.query_image(embed_text_clip(window_text), top_k=top_k)

            seen: set[str] = set()
            for candidate_id, similarity in list(text_hits) + list(image_hits):
                if candidate_id in window_id_set or candidate_id in seen:
                    continue
                seen.add(candidate_id)
                candidate = by_id.get(candidate_id)
                if candidate is None or candidate.modality in SPEECH_MODALITIES:
                    continue
                if similarity < config.SEMANTIC_LINK_MIN_SIMILARITY:
                    continue

                label, label_confidence = provider.classify_link(
                    "speech", window_text, candidate.modality.value, candidate.content
                )
                if label is None:
                    continue
                confidence = max(similarity, label_confidence)
                for unit in window_units:
                    if _record_link(unit, candidate, "semantic_similarity", confidence, label, sqlite, graph):
                        created += 1
    return created


def link_same_entity(units: list[EvidenceUnit], *, sqlite: SQLiteStore, graph: GraphStore) -> int:
    by_entity: dict[str, list[EvidenceUnit]] = {}
    for u in units:
        for e in u.entities:
            by_entity.setdefault(e.text, []).append(u)

    created = 0
    for entity_name, group in by_entity.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.source_file == b.source_file and a.modality == b.modality:
                    continue  # not interesting: same file, same modality co-mention
                if _record_link(a, b, "same_entity", 0.9, None, sqlite, graph):
                    created += 1
    return created


def run_linking(*, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore) -> dict:
    units = sqlite.all()
    temporal = link_temporal(units, sqlite=sqlite, graph=graph)
    semantic = link_semantic(units, vectors=vectors, sqlite=sqlite, graph=graph)
    same_entity = link_same_entity(units, sqlite=sqlite, graph=graph)
    logger.info(
        "Linking complete: %d temporal, %d semantic, %d same-entity links", temporal, semantic, same_entity
    )
    return {"temporal_links": temporal, "semantic_links": semantic, "same_entity_links": same_entity}
