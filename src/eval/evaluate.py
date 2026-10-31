"""
Evaluates the multimodal system against the naive text-only baseline on the
curated query set in queries.py, reporting keyword recall (a cheap proxy for
"did we retrieve the right content") and modality coverage — the metric that
operationalizes the actual hackathon challenge (cross-modal retrieval, not
generic text recall). The baseline is structurally incapable of surfacing
non-text modalities (video_frame/image/pdf_image); its modality coverage is
computed accordingly rather than measured against evidence it can't have.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from src.baseline.text_only_rag import run_baseline_query
from src.eval.queries import EVAL_QUERIES
from src.retrieval.query_engine import run_query
from src.storage.graph_store import GraphStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.vector_store import VectorStore

TEXTUAL_ACHIEVABLE_MODALITIES = {"transcript_segment", "pdf_text", "ocr_text"}


def _keyword_recall(keywords: list[str], texts: list[str]) -> float:
    if not keywords:
        return 1.0
    haystack = " ".join(texts).lower()
    hits = sum(1 for kw in keywords if kw.lower() in haystack)
    return hits / len(keywords)


def _modality_coverage(expected: list[str], found_modalities: set[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for m in expected if m in found_modalities)
    return hits / len(expected)


def run_eval() -> dict:
    sqlite, vectors, graph = SQLiteStore(), VectorStore(), GraphStore()
    rows = []

    for spec in EVAL_QUERIES:
        mm_result = run_query(spec["query"], sqlite=sqlite, vectors=vectors, graph=graph)
        mm_texts = [e["content"] for e in mm_result.get("evidence", [])]
        mm_modalities = {e["modality"] for e in mm_result.get("evidence", [])}

        bl_result = run_baseline_query(spec["query"], sqlite=sqlite)
        bl_texts = bl_result.get("chunks_used", [])
        baseline_can_cover_modalities = set(spec["expected_modalities"]) <= TEXTUAL_ACHIEVABLE_MODALITIES

        rows.append({
            "id": spec["id"], "category": spec["category"], "query": spec["query"],
            "multimodal": {
                "keyword_recall": _keyword_recall(spec["expected_keywords"], mm_texts),
                "modality_coverage": _modality_coverage(spec["expected_modalities"], mm_modalities),
                "modalities_found": sorted(mm_modalities),
                "evidence_count": len(mm_texts),
                "answer": mm_result.get("answer", ""),
            },
            "baseline": {
                "keyword_recall": _keyword_recall(spec["expected_keywords"], bl_texts),
                "modality_coverage": 1.0 if baseline_can_cover_modalities else 0.0,
                "answer": bl_result.get("answer", ""),
            },
        })

    def avg(system: str, metric: str) -> float:
        vals = [r[system][metric] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "multimodal_avg_keyword_recall": avg("multimodal", "keyword_recall"),
        "multimodal_avg_modality_coverage": avg("multimodal", "modality_coverage"),
        "baseline_avg_keyword_recall": avg("baseline", "keyword_recall"),
        "baseline_avg_modality_coverage": avg("baseline", "modality_coverage"),
    }
    report = {"summary": summary, "rows": rows}

    out_path = Path(config.ROOT_DIR) / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report


def print_report(report: dict) -> None:
    s = report["summary"]
    print("=== Multimodal vs. Text-Only Baseline ===")
    print(f"Keyword recall     — multimodal: {s['multimodal_avg_keyword_recall']:.2f}  baseline: {s['baseline_avg_keyword_recall']:.2f}")
    print(f"Modality coverage  — multimodal: {s['multimodal_avg_modality_coverage']:.2f}  baseline: {s['baseline_avg_modality_coverage']:.2f}")
    print()
    for row in report["rows"]:
        print(f"[{row['category']:>13}] {row['id']}")
        mm = row["multimodal"]
        print(f"  multimodal: recall={mm['keyword_recall']:.2f} modality_cov={mm['modality_coverage']:.2f} modalities={mm['modalities_found']}")
        print(f"  baseline:   recall={row['baseline']['keyword_recall']:.2f} modality_cov={row['baseline']['modality_coverage']:.2f}")


if __name__ == "__main__":
    print_report(run_eval())
