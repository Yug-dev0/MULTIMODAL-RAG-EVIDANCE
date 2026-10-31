import os
from dataclasses import dataclass
import fitz  # PyMuPDF


@dataclass
class PDFPage:
    page_number: int
    text: str
    image_paths: list[str]


def extract_pdf(pdf_path: str, out_dir: str) -> list[PDFPage]:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text()
        image_paths = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_path = os.path.join(out_dir, f"page{page_index + 1}_img{img_index}.png")
            pix.save(image_path)
            image_paths.append(image_path)
        pages.append(PDFPage(page_number=page_index + 1, text=text, image_paths=image_paths))
    return pages
