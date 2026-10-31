from app.retrieval.fusion_rerank import fuse_and_rerank


def test_fuse_and_rerank_orders_by_weighted_score():
    ranked = fuse_and_rerank(
        vector_hits={"a": 0.9, "b": 0.3},
        graph_hit_ids={"b"},
        entity_hit_ids={"a", "b"},
        confidences={"a": 1.0, "b": 1.0},
    )
    ids = [r.id for r in ranked]
    assert set(ids) == {"a", "b"}
    assert ranked[0].score >= ranked[1].score


def test_fuse_and_rerank_downweights_superseded_evidence():
    ranked = fuse_and_rerank(
        vector_hits={"old": 0.9, "new": 0.9},
        graph_hit_ids=set(), entity_hit_ids=set(),
        confidences={"old": 1.0, "new": 1.0},
        superseded_by={"old": "new"},
    )
    old_item = next(r for r in ranked if r.id == "old")
    new_item = next(r for r in ranked if r.id == "new")
    assert old_item.superseded is True
    assert old_item.score < new_item.score
