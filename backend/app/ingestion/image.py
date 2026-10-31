import uuid
from app.extraction.ocr import ocr_image
from app.extraction.vision import caption_image
from app.extraction.embeddings import embed_texts  # noqa: F401
from app.storage.repository import persist_evidence_unit


async def ingest_image(session, image_path: str, source_id: uuid.UUID) -> list:
    units = []
    full_text, mean_conf, _words = ocr_image(image_path)
    if full_text.strip():
        units.append(await persist_evidence_unit(
            session, content_text=full_text, content_type="ocr", modality="ocr",
            source_id=source_id, source_type="image", raw_locator=image_path,
            extraction_method="tesseract", confidence=mean_conf,
        ))
    caption = await caption_image(image_path)
    units.append(await persist_evidence_unit(
        session, content_text=caption.description, content_type="caption", modality="image",
        source_id=source_id, source_type="image", raw_locator=image_path,
        extraction_method="gpt-4o-mini-vision", confidence=caption.confidence,
        metadata_json={"caption": caption.caption},
    ))
    return units
