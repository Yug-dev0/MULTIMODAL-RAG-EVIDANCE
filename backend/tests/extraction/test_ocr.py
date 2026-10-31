import os
from app.extraction.ocr import ocr_image

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "ocr_sample.png")


def test_ocr_extracts_text_with_confidence():
    full_text, mean_conf, words = ocr_image(FIXTURE)
    assert len(full_text.strip()) > 0
    assert 0.0 <= mean_conf <= 1.0
    assert len(words) > 0
    assert all(len(w.bbox) == 4 for w in words)
