"""
Central configuration. Every tunable threshold/model name/path lives here —
no magic numbers scattered through the pipeline. Override any value via
environment variables (see .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# override=True: a stale placeholder GOOGLE_API_KEY has been observed already
# exported in the shell environment on this machine, which silently shadows
# the real value in .env (python-dotenv does not override existing env vars
# by default). .env is the intended source of truth for this project.
load_dotenv(override=True)

ROOT_DIR = Path(__file__).parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SQLITE_PATH = ROOT_DIR / "db" / "sqlite" / "evidence.db"
CHROMA_DIR = ROOT_DIR / "db" / "chroma"
KUZU_DIR = ROOT_DIR / "db" / "kuzu"

# --- LLM provider ---
# "ollama" (local, default — no external dependency) or "gemini" (requires a
# working GOOGLE_API_KEY from https://aistudio.google.com/apikey). Swapping
# this is the only change needed to move between local and hosted inference;
# see src/enrichment/llm_provider.py.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "qwen2.5vl:7b")

# --- ASR ---
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")  # "cpu" | "cuda"; MPS not supported by faster-whisper
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# --- Scene detection / keyframes ---
SCENE_DETECT_THRESHOLD = float(os.environ.get("SCENE_DETECT_THRESHOLD", "27.0"))
MIN_SCENE_LEN_SECONDS = float(os.environ.get("MIN_SCENE_LEN_SECONDS", "1.5"))

# --- OCR ---
OCR_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "55.0"))  # tesseract 0-100 scale

# --- Embeddings ---
TEXT_EMBEDDING_MODEL = os.environ.get("TEXT_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "mps")  # falls back to cpu automatically if unavailable

# --- Cross-modal linking ---
TEMPORAL_COOCCURRENCE_WINDOW_SECONDS = float(os.environ.get("TEMPORAL_WINDOW_SECONDS", "5.0"))
SEMANTIC_LINK_MIN_SIMILARITY = float(os.environ.get("SEMANTIC_LINK_MIN_SIMILARITY", "0.25"))
ENTITY_COREF_MIN_SIMILARITY = float(os.environ.get("ENTITY_COREF_MIN_SIMILARITY", "0.82"))

# --- Retrieval ---
RETRIEVAL_TOP_K_TEXT = int(os.environ.get("RETRIEVAL_TOP_K_TEXT", "8"))
RETRIEVAL_TOP_K_IMAGE = int(os.environ.get("RETRIEVAL_TOP_K_IMAGE", "5"))
GRAPH_TRAVERSAL_HOPS = int(os.environ.get("GRAPH_TRAVERSAL_HOPS", "2"))
FUSION_WEIGHT_VECTOR = float(os.environ.get("FUSION_WEIGHT_VECTOR", "0.5"))
FUSION_WEIGHT_GRAPH = float(os.environ.get("FUSION_WEIGHT_GRAPH", "0.3"))
FUSION_WEIGHT_CONFIDENCE = float(os.environ.get("FUSION_WEIGHT_CONFIDENCE", "0.2"))
FINAL_TOP_K = int(os.environ.get("FINAL_TOP_K", "8"))

# --- Baseline (text-only RAG) ---
BASELINE_CHUNK_SIZE_CHARS = int(os.environ.get("BASELINE_CHUNK_SIZE_CHARS", "500"))
BASELINE_CHUNK_OVERLAP_CHARS = int(os.environ.get("BASELINE_CHUNK_OVERLAP_CHARS", "50"))
BASELINE_TOP_K = int(os.environ.get("BASELINE_TOP_K", "5"))

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, SQLITE_PATH.parent, CHROMA_DIR, KUZU_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
