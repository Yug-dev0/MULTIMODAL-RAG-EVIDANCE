import asyncio
import uuid
from app.extraction.pdf import extract_pdf
from app.extraction.ocr import ocr_image
from app.extraction.vision import caption_image
from app.extraction.embeddings import embed_texts  # noqa: F401
from app.storage.repository import persist_evidence_unit


async def ingest_pdf(session, pdf_path: str, source_id: uuid.UUID, media_dir: str) -> list:
    pages = extract_pdf(pdf_path, media_dir)
    units = []

    # Flatten every embedded image across all pages and caption them all
    # concurrently (bounded by LLM_SEMAPHORE) instead of one page/image at a
    # time — a multi-page deck with several diagrams per page previously paid
    # for each caption call's network latency back to back.
    image_refs = [(page, image_path) for page in pages for image_path in page.image_paths]
    captions = await asyncio.gather(*(caption_image(image_path) for _page, image_path in image_refs))
    captions_by_ref = {image_path: caption for (_page, image_path), caption in zip(image_refs, captions)}

    for page in pages:
        if page.text.strip():
            units.append(await persist_evidence_unit(
                session, content_text=page.text, content_type="document_text", modality="document",
                source_id=source_id, source_type="pdf", raw_locator=f"{pdf_path}#page={page.page_number}",
                extraction_method="pymupdf", page_number=page.page_number, confidence=0.98,
            ))
        for image_path in page.image_paths:
            full_text, mean_conf, _words = ocr_image(image_path)
            if full_text.strip():
                units.append(await persist_evidence_unit(
                    session, content_text=full_text, content_type="ocr", modality="ocr",
                    source_id=source_id, source_type="pdf", raw_locator=image_path,
                    extraction_method="tesseract", page_number=page.page_number, confidence=mean_conf,
                ))
            caption = captions_by_ref[image_path]
            units.append(await persist_evidence_unit(
                session, content_text=caption.description, content_type="caption", modality="image",
                source_id=source_id, source_type="pdf", raw_locator=image_path,
                extraction_method="gpt-4o-mini-vision", page_number=page.page_number,
                confidence=caption.confidence, metadata_json={"caption": caption.caption},
            ))
    return units
