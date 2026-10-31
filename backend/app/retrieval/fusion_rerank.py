from dataclasses import dataclass


@dataclass
class RankedItem:
    id: str
    score: float
    superseded: bool = False


def fuse_and_rerank(*, vector_hits: dict, graph_hit_ids: set, entity_hit_ids: set,
                     confidences: dict, superseded_by: dict | None = None) -> list[RankedItem]:
    superseded_by = superseded_by or {}
    all_ids = set(vector_hits) | graph_hit_ids | entity_hit_ids
    items = []
    for item_id in all_ids:
        base = (0.5 * vector_hits.get(item_id, 0.0)
                + 0.3 * (1.0 if item_id in graph_hit_ids else 0.0)
                + 0.2 * (1.0 if item_id in entity_hit_ids else 0.0))
        score = base * confidences.get(item_id, 1.0)
        is_superseded = item_id in superseded_by
        if is_superseded:
            score *= 0.5
        items.append(RankedItem(id=item_id, score=score, superseded=is_superseded))
    items.sort(key=lambda r: r.score, reverse=True)
    return items
