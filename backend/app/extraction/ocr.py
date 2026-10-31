from dataclasses import dataclass
import pytesseract
from PIL import Image


@dataclass
class OCRWord:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float


def ocr_image(image_path: str) -> tuple[str, float, list[OCRWord]]:
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        words.append(OCRWord(
            text=text,
            bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
            confidence=conf / 100.0,
        ))
    full_text = " ".join(w.text for w in words)
    mean_conf = sum(w.confidence for w in words) / len(words) if words else 0.0
    return full_text, mean_conf, words
