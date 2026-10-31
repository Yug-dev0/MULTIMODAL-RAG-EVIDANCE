import uuid
import pytest
from unittest.mock import patch
from app.storage.db import get_session
from app.ingestion.image import ingest_image
from app.extraction.ocr import OCRWord
from app.extraction.vision import ImageCaption

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_ingest_image_creates_ocr_and_caption_units():
    source_id = uuid.uuid4()
    with (
        patch("app.ingestion.image.ocr_image", return_value=("DIAGRAM TITLE", 0.8, [OCRWord("DIAGRAM", (0,0,5,5), 0.8)])),
        patch("app.ingestion.image.caption_image", return_value=ImageCaption("Diagram", "A network diagram.", 0.7)),
        patch("app.storage.repository.embed_texts", return_value=[[0.1] * 1536]),
    ):
        async with get_session() as session:
            units = await ingest_image(session, "diagram.png", source_id)
            await session.commit()
    assert sorted(u.modality for u in units) == ["image", "ocr"]
    assert all(u.source_type == "image" for u in units)
