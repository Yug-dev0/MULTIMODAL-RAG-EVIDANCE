from run_eval import compute_modality_recall, format_report


def test_compute_modality_recall_full_match():
    assert compute_modality_recall({"speech", "ocr"}, {"speech", "ocr", "image"}) == 1.0


def test_compute_modality_recall_partial_match():
    assert compute_modality_recall({"speech", "ocr"}, {"speech"}) == 0.5


def test_compute_modality_recall_empty_requirement_is_perfect():
    assert compute_modality_recall(set(), {"speech"}) == 1.0


def test_format_report_includes_average_and_query_id():
    report = format_report([
        {"id": "q1", "query": "x", "multimodal_recall": 1.0, "multimodal_answer": "a", "baseline_answer": "b"},
    ])
    assert "Average modality recall" in report
    assert "q1" in report
