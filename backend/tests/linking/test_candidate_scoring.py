import uuid
from app.storage.models import EvidenceUnit
from app.linking.candidate_scoring import score_candidates


def _unit(text, start, end, embedding, modality="speech"):
    return EvidenceUnit(
        id=uuid.uuid4(), content_text=text, content_type="x", modality=modality,
        source_id=uuid.uuid4(), source_type="video", raw_locator="v", extraction_method="x",
        timestamp_start=start, timestamp_end=end, embedding=embedding, confidence=0.9, metadata_json={},
    )


def test_score_candidates_ranks_closer_more_similar_candidate_higher():
    window_units = [_unit("we should use redis architecture for caching", 10.0, 15.0, [1.0, 0.0])]
    good = _unit("REDIS ARCHITECTURE DIAGRAM", 12.0, 12.0, [0.95, 0.05], modality="ocr")
    bad = _unit("unrelated quarterly sales numbers", 90.0, 90.0, [0.0, 1.0], modality="ocr")
    ranked = score_candidates(window_units, [bad, good], top_k=3)
    assert ranked[0][0] is good
    assert ranked[0][1] > ranked[1][1]


def test_score_candidates_respects_top_k():
    window_units = [_unit("topic text", 0.0, 5.0, [1.0, 0.0])]
    candidates = [_unit(f"candidate {i}", float(i), float(i), [1.0, 0.0], modality="ocr") for i in range(5)]
    ranked = score_candidates(window_units, candidates, top_k=2)
    assert len(ranked) == 2
