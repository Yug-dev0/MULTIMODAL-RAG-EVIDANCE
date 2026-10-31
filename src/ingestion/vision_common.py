"""
Shared vision helpers used by the video, image, and PDF adapters so all
three extract on-screen/embedded text and diagram understanding the same
way, rather than reimplementing OCR/captioning three times.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from PIL import Image

import config
from src.enrichment.llm_provider import get_llm_provider


def _bbox_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def _contains_frac(inner: tuple, outer: tuple) -> float:
    """Fraction of `inner` rect (x,y,w,h) area covered by `outer` rect."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    ax1, ay1 = max(ix, ox), max(iy, oy)
    ax2, ay2 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    if ax2 <= ax1 or ay2 <= ay1:
        return 0.0
    inter = (ax2 - ax1) * (ay2 - ay1)
    return inter / (iw * ih) if iw * ih > 0 else 0.0


def _detect_box_regions(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Finds bordered rectangular panels (diagram boxes) via edge/contour
    detection. Tesseract's own layout analysis frequently classifies bordered
    boxes as non-text "image" regions and skips OCR on their interior — this
    recovers those regions so they get OCR'd directly.
    """
    h, w = gray.shape
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    img_area = w * h
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.0015 * img_area or area > 0.45 * img_area:
            continue
        aspect = cw / max(ch, 1)
        if aspect < 0.25 or aspect > 6:
            continue
        candidates.append((x, y, cw, ch))

    # Drop contours that are merge artifacts (e.g. a box + connecting arrow +
    # neighboring box all traced as one blob) — a real single label box
    # shouldn't fully contain two or more other distinct candidate boxes.
    boxes = []
    for i, box in enumerate(candidates):
        contained = [
            other for j, other in enumerate(candidates)
            if j != i and other != box and _contains_frac(other, box) > 0.85
        ]
        if len(contained) >= 2:
            continue
        boxes.append(box)

    # NMS-style de-dup of near-identical/nested boxes from double edges.
    boxes.sort(key=lambda b: b[2] * b[3])
    deduped: list[tuple[int, int, int, int]] = []
    for b in boxes:
        bx, by, bw, bh = b
        b_xyxy = (bx, by, bx + bw, by + bh)
        if any(_bbox_iou(
            {"x": b[0], "y": b[1], "width": b[2], "height": b[3]},
            {"x": d[0], "y": d[1], "width": d[2], "height": d[3]},
        ) > 0.6 for d in deduped):
            continue
        deduped.append(b)
    return deduped


def _ocr_lines(img: Image.Image, offset: tuple[int, int] = (0, 0), psm: int = 6) -> list[dict]:
    """OCR one crop, grouped into lines, with bboxes translated to full-image coords."""
    ox, oy = offset
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=f"--psm {psm}")
    lines: dict[tuple, dict] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        l, t, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        entry = lines.setdefault(key, {"words": [], "confs": [], "x1": l, "y1": t, "x2": l + ww, "y2": t + hh})
        entry["words"].append(word)
        entry["confs"].append(conf)
        entry["x1"] = min(entry["x1"], l)
        entry["y1"] = min(entry["y1"], t)
        entry["x2"] = max(entry["x2"], l + ww)
        entry["y2"] = max(entry["y2"], t + hh)

    out = []
    for entry in lines.values():
        out.append({
            "text": " ".join(entry["words"]),
            "confidence": (sum(entry["confs"]) / len(entry["confs"])) / 100.0,
            "bbox": {
                "x": float(ox + entry["x1"]),
                "y": float(oy + entry["y1"]),
                "width": float(entry["x2"] - entry["x1"]),
                "height": float(entry["y2"] - entry["y1"]),
            },
        })
    return out


def run_ocr_regions(image_path: str, box_pad: int = 8) -> list[dict]:
    """
    Region-aware OCR with line-level bounding boxes (in full-image pixel
    coords). Combines two passes:

      1. Contour-based box detection (`_detect_box_regions`) + OCR on each
         detected panel's interior — recovers diagram/label text inside
         bordered boxes that a whole-image pass tends to skip entirely
         (Tesseract's layout analysis often classifies a bordered rectangle
         as a non-text "image" region).
      2. A whole-image OCR pass for free-floating text (titles, captions,
         prose) that isn't inside any detected box.

    Box-pass results take precedence inside their own regions; whole-image
    results are only added where they don't overlap an already-found box
    region, avoiding duplicate/garbled overlap between the two passes.

    Returns a list of {"text", "confidence" (0-1), "bbox": {x,y,width,height}}.
    """
    img = Image.open(image_path).convert("RGB")
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    w, h = img.size

    box_regions: list[dict] = []
    for bx, by, bw, bh in _detect_box_regions(gray):
        x1, y1 = max(bx + box_pad, 0), max(by + box_pad, 0)
        x2, y2 = min(bx + bw - box_pad, w), min(by + bh - box_pad, h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((x1, y1, x2, y2))
        box_regions.extend(_ocr_lines(crop, offset=(x1, y1), psm=6))

    whole_regions = _ocr_lines(img, offset=(0, 0), psm=3)

    merged = list(box_regions)
    for r in whole_regions:
        if not any(_bbox_iou(r["bbox"], b["bbox"]) > 0.15 for b in box_regions):
            merged.append(r)

    deduped: list[dict] = []
    for r in sorted(merged, key=lambda r: -r["confidence"]):
        if any(r["text"].lower() == d["text"].lower() and _bbox_iou(r["bbox"], d["bbox"]) > 0.3 for d in deduped):
            continue
        deduped.append(r)
    return deduped


def run_ocr(image_path: str) -> tuple[str, float]:
    """Convenience wrapper: full concatenated text + mean confidence in [0,1]."""
    regions = run_ocr_regions(image_path)
    if not regions:
        return "", 0.0
    text = " ".join(r["text"] for r in regions)
    mean_conf = sum(r["confidence"] for r in regions) / len(regions)
    return text, mean_conf


def caption_image(image_path: str, context: str = "") -> tuple[str, float]:
    """Returns (caption, confidence in [0,1]) via the configured LLM provider."""
    provider = get_llm_provider()
    return provider.caption_image(image_path, context=context)


def is_substantive_ocr(text: str, min_chars: int = 3) -> bool:
    return len(text.strip()) >= min_chars
