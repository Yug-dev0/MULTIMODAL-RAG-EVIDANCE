"""
Safe-fallback entity resolution against the Kuzu graph: exact/case-insensitive
match, then embedding-similarity match above config.ENTITY_COREF_MIN_SIMILARITY,
then create a new canonical entity. Used by every ingestion adapter so the
same real-world thing ("Postgres" / "PostgreSQL" / "the database") resolves
to one canonical node no matter which file or modality it was mentioned in.
"""

from __future__ import annotations

import config
from src.enrichment.embeddings import cosine_similarity, embed_text
from src.storage.graph_store import GraphStore

_embedding_cache: dict[str, list[float]] = {}


def _cached_embed(name: str) -> list[float]:
    if name not in _embedding_cache:
        _embedding_cache[name] = embed_text(name)
    return _embedding_cache[name]


def resolve_entity_name(graph: GraphStore, mention_text: str, mention_type: str, threshold: float | None = None) -> str:
    """Returns the canonical entity name to use, creating a graph node for it if genuinely new."""
    threshold = config.ENTITY_COREF_MIN_SIMILARITY if threshold is None else threshold
    normalized = mention_text.strip()
    if not normalized:
        return normalized

    existing = graph.all_entity_names()
    for name, _type in existing:
        if name.strip().lower() == normalized.lower():
            return name

    if existing:
        mention_embedding = _cached_embed(normalized)
        best_name, best_similarity = None, 0.0
        for name, _type in existing:
            similarity = cosine_similarity(mention_embedding, _cached_embed(name))
            if similarity > best_similarity:
                best_name, best_similarity = name, similarity
        if best_name is not None and best_similarity >= threshold:
            return best_name

    graph.upsert_entity_node(normalized, mention_type)
    return normalized


def resolve_entity_name_readonly(graph: GraphStore, mention_text: str, threshold: float | None = None) -> str | None:
    """
    Same matching logic as resolve_entity_name, but never creates a new node —
    used at query time, where a query mentioning something never seen in the
    corpus should simply fail to match rather than pollute the graph.
    """
    threshold = config.ENTITY_COREF_MIN_SIMILARITY if threshold is None else threshold
    normalized = mention_text.strip()
    if not normalized:
        return None

    existing = graph.all_entity_names()
    for name, _type in existing:
        if name.strip().lower() == normalized.lower():
            return name

    if existing:
        mention_embedding = _cached_embed(normalized)
        best_name, best_similarity = None, 0.0
        for name, _type in existing:
            similarity = cosine_similarity(mention_embedding, _cached_embed(name))
            if similarity > best_similarity:
                best_name, best_similarity = name, similarity
        if best_name is not None and best_similarity >= threshold:
            return best_name

    return None
