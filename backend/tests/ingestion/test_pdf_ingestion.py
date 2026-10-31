import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.ingestion.pdf import ingest_pdf
from app.extraction.pdf import PDFPage
from app.extraction.ocr import OCRWord
from app.extraction.vision import ImageCaption

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_ingest_pdf_creates_document_text_and_image_units(tmp_path):
    source_id = uuid.uuid4()
    with (
        patch("app.ingestion.pdf.extract_pdf", return_value=[
            PDFPage(page_number=1, text="Redis reduces load on Postgres.", image_paths=["fig1.png"]),
        ]),
        patch("app.ingestion.pdf.ocr_image", return_value=("FIG 1: CACHE LAYER", 0.75, [OCRWord("FIG", (0,0,5,5), 0.75)])),
        patch("app.ingestion.pdf.caption_image", return_value=ImageCaption("Cache layer figure", "A figure showing a cache layer.", 0.7)),
        patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]),
    ):
        async with get_session() as session:
            units = await ingest_pdf(session, "doc.pdf", source_id, str(tmp_path))
            await session.commit()
    content_types = sorted(u.content_type for u in units)
    assert content_types == ["caption", "document_text", "ocr"]
    doc_unit = next(u for u in units if u.content_type == "document_text")
    assert doc_unit.page_number == 1
