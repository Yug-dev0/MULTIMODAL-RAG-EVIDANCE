"""
Two embedding spaces, deliberately kept separate:

  - `embed_text`: sentence-transformers, used for transcript/OCR/PDF text
    content in the `text_embeddings` Chroma collection.
  - `embed_image_clip` / `embed_text_clip`: CLIP's image and text towers,
    sharing one space. This is what makes real cross-modal retrieval
    possible — a text query embedded with `embed_text_clip` can be compared
    directly against frame/image vectors produced by `embed_image_clip`,
    independent of any OCR/caption text overlap.

Both are lazily loaded singletons so importing this module is cheap; the
actual model weights only load on first use.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

import config

logger = logging.getLogger(__name__)

_text_model = None
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _resolve_device(requested: str) -> str:
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_text_model():
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer

        device = _resolve_device(config.EMBEDDING_DEVICE)
        logger.info("Loading text embedding model %s on %s", config.TEXT_EMBEDDING_MODEL, device)
        _text_model = SentenceTransformer(config.TEXT_EMBEDDING_MODEL, device=device)
    return _text_model


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip

        device = _resolve_device(config.EMBEDDING_DEVICE)
        arch, pretrained = _clip_arch_and_pretrained(config.CLIP_MODEL_NAME)
        logger.info("Loading CLIP model %s (%s) on %s", arch, pretrained, device)
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(arch)
        model.eval()
        _clip_model, _clip_preprocess, _clip_tokenizer = model, preprocess, tokenizer
        _clip_model._device = device
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _clip_arch_and_pretrained(name: str) -> tuple[str, str]:
    """Maps a HF-style CLIP name to an open_clip (arch, pretrained) pair."""
    known = {
        "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
        "openai/clip-vit-base-patch16": ("ViT-B-16", "openai"),
        "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
    }
    return known.get(name, ("ViT-B-32", "openai"))


def embed_text(text: str) -> list[float]:
    model = _get_text_model()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_text_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_image_clip(image_path: str) -> list[float]:
    from PIL import Image

    model, preprocess, _ = _get_clip()
    device = getattr(model, "_device", "cpu")
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().tolist()


def embed_text_clip(text: str) -> list[float]:
    model, _, tokenizer = _get_clip()
    device = getattr(model, "_device", "cpu")
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(np.dot(va, vb) / denom)
