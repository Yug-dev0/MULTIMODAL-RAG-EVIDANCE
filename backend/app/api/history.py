import time
from dataclasses import dataclass, field

MAX_HISTORY = 200


@dataclass
class QueryLogEntry:
    query: str
    mode: str  # "query" | "compare"
    citation_count: int
    modality_count: int
    baseline_citation_count: int
    timestamp: float = field(default_factory=time.time)


QUERY_LOG: list[QueryLogEntry] = []


def record_query(entry: QueryLogEntry) -> None:
    QUERY_LOG.append(entry)
    del QUERY_LOG[:-MAX_HISTORY]


def recent_queries(limit: int = 10) -> list[QueryLogEntry]:
    return list(reversed(QUERY_LOG[-limit:]))


def session_summary() -> dict:
    """Live, session-based comparison metrics -- derived only from actual
    /compare calls made so far. No fabricated percentages: if nothing has
    run yet, the counts are honestly zero."""
    compare_entries = [e for e in QUERY_LOG if e.mode == "compare"]
    total = len(compare_entries)
    if total == 0:
        return {
            "comparisons_run": 0,
            "avg_multimodal_citations": 0.0,
            "avg_baseline_citations": 0.0,
            "queries_with_cross_modal_evidence": 0,
            "queries_baseline_had_no_provenance": 0,
        }
    avg_multimodal = sum(e.citation_count for e in compare_entries) / total
    avg_baseline = sum(e.baseline_citation_count for e in compare_entries) / total
    cross_modal = sum(1 for e in compare_entries if e.modality_count > 1)
    no_provenance = sum(1 for e in compare_entries if e.baseline_citation_count == 0)
    return {
        "comparisons_run": total,
        "avg_multimodal_citations": round(avg_multimodal, 2),
        "avg_baseline_citations": round(avg_baseline, 2),
        "queries_with_cross_modal_evidence": cross_modal,
        "queries_baseline_had_no_provenance": no_provenance,
    }
