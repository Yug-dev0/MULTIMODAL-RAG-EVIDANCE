import os
from app.extraction.pdf import extract_pdf

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample.pdf")


def test_extract_pdf_returns_pages_with_text(tmp_path):
    pages = extract_pdf(FIXTURE, str(tmp_path))
    assert len(pages) >= 1
    assert pages[0].page_number == 1
    assert isinstance(pages[0].text, str)
