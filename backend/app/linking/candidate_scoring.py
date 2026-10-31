from app.linking.scoring_utils import cosine_similarity


def _mean_embedding(units) -> list[float]:
    dim = len(units[0].embedding)
    sums = [0.0] * dim
    for u in units:
        for i, v in enumerate(u.embedding):
            sums[i] += v
    return [s / len(units) for s in sums]


def _keyword_overlap(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _timestamp_proximity(window_units, candidate, norm: float) -> float:
    start = min(u.timestamp_start for u in window_units)
    end = max(u.timestamp_end for u in window_units)
    t = candidate.timestamp_start
    distance = 0.0 if start <= t <= end else min(abs(t - start), abs(t - end))
    return 1.0 - min(1.0, distance / norm)


def score_candidates(window_units, candidates, top_k: int = 3, timestamp_norm: float = 15.0):
    window_text = " ".join(u.content_text for u in window_units)
    window_embedding = _mean_embedding(window_units)
    scored = []
    for candidate in candidates:
        timestamp_proximity = _timestamp_proximity(window_units, candidate, timestamp_norm)
        keyword_overlap = _keyword_overlap(window_text, candidate.content_text)
        embedding_similarity = cosine_similarity(window_embedding, candidate.embedding)
        score = 0.35 * timestamp_proximity + 0.30 * keyword_overlap + 0.35 * embedding_similarity
        scored.append((candidate, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
