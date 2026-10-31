import uuid
from app.storage.models import EvidenceUnit
from app.linking.scoring_utils import cosine_similarity
from app.linking.segmentation import segment_into_windows


def _unit(start, end, speaker, embedding):
    return EvidenceUnit(
        id=uuid.uuid4(), content_text="x", content_type="transcript", modality="speech",
        source_id=uuid.uuid4(), source_type="video", raw_locator="v", extraction_method="whisper-1",
        timestamp_start=start, timestamp_end=end, embedding=embedding, confidence=0.9,
        metadata_json={"speaker": speaker},
    )


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_segment_merges_same_speaker_similar_content_no_gap():
    units = [
        _unit(0.0, 2.0, "SPEAKER_00", [1.0, 0.0]),
        _unit(2.0, 4.0, "SPEAKER_00", [0.99, 0.01]),
    ]
    windows = segment_into_windows(units)
    assert len(windows) == 1
    assert windows[0].evidence_unit_ids == [units[0].id, units[1].id]


def test_segment_splits_on_speaker_change_and_topic_shift():
    units = [
        _unit(0.0, 2.0, "SPEAKER_00", [1.0, 0.0]),
        _unit(2.0, 4.0, "SPEAKER_01", [0.0, 1.0]),
    ]
    windows = segment_into_windows(units)
    assert len(windows) == 2
