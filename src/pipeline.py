"""
CLI orchestrator: ingest one or more sources (video/audio/image/pdf,
auto-detected from extension), then run cross-modal linking and event
assembly over the whole corpus. Ingestion is idempotent per source file name
(tracked in the `source_files` table), so re-running the same file is a
no-op unless --force is passed.

Usage:
    python -m src.pipeline ingest test_data/tech_talk.mp4 test_data/design_notes.pdf --link
    python -m src.pipeline link
    python -m src.pipeline stats
    python -m src.pipeline reset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import config
from src.enrichment.cross_modal_linker import run_linking
from src.enrichment.event_assembly import assemble_events
from src.ingestion.audio import ingest_audio
from src.ingestion.image import ingest_image
from src.ingestion.pdf import ingest_pdf
from src.ingestion.video import ingest_video
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PDF_EXTENSIONS = {".pdf"}


def ingest_source(
    path: str, *, sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore, force: bool = False
) -> int:
    p = Path(path)
    source_file = p.name
    already = {row["source_file"] for row in sqlite.source_files()}
    if source_file in already and not force:
        logger.info("Skipping %s (already ingested; pass --force to redo)", source_file)
        return 0

    ext = p.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        units = ingest_video(str(p), sqlite=sqlite, vectors=vectors, graph=graph)
    elif ext in AUDIO_EXTENSIONS:
        units = ingest_audio(str(p), sqlite=sqlite, vectors=vectors, graph=graph)
    elif ext in IMAGE_EXTENSIONS:
        units = ingest_image(str(p), sqlite=sqlite, vectors=vectors, graph=graph)
    elif ext in PDF_EXTENSIONS:
        units = ingest_pdf(str(p), sqlite=sqlite, vectors=vectors, graph=graph)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return len(units)


def _link_and_assemble(sqlite: SQLiteStore, vectors: VectorStore, graph: GraphStore) -> None:
    stats = run_linking(sqlite=sqlite, vectors=vectors, graph=graph)
    logger.info("Linking: %s", stats)
    events = assemble_events(sqlite=sqlite, vectors=vectors, graph=graph)
    logger.info("Assembled %d multimodal knowledge events", len(events))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal RAG-ready pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest one or more source files")
    ingest_p.add_argument("paths", nargs="+")
    ingest_p.add_argument("--force", action="store_true")
    ingest_p.add_argument("--link", action="store_true", help="Also run linking + event assembly afterward")

    sub.add_parser("link", help="Run cross-modal linking + event assembly over the whole corpus")
    sub.add_parser("reset", help="Wipe SQLite + Chroma and start fresh")
    sub.add_parser("stats", help="Print corpus stats")

    args = parser.parse_args()

    sqlite = SQLiteStore()
    vectors = VectorStore()
    graph = GraphStore()

    if args.command == "reset":
        sqlite.clear()
        vectors.reset()
        logger.info("Cleared SQLite + Chroma. Kuzu graph at %s was left as-is.", config.KUZU_DIR)
        return

    if args.command == "stats":
        units = sqlite.all()
        by_modality: dict[str, int] = {}
        for u in units:
            by_modality[u.modality.value] = by_modality.get(u.modality.value, 0) + 1
        print(f"Sources: {len(sqlite.source_files())}")
        print(f"Evidence units: {len(units)}")
        for modality, count in sorted(by_modality.items()):
            print(f"  {modality}: {count}")
        print(f"Knowledge events: {len(sqlite.all_events())}")
        return

    if args.command == "ingest":
        total = 0
        for path in args.paths:
            total += ingest_source(path, sqlite=sqlite, vectors=vectors, graph=graph, force=args.force)
        logger.info("Ingested %d new evidence units total", total)
        if args.link:
            _link_and_assemble(sqlite, vectors, graph)
        return

    if args.command == "link":
        _link_and_assemble(sqlite, vectors, graph)
        return


if __name__ == "__main__":
    main()
