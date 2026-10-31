"""
Hand-authored evaluation queries. This is a test fixture, not pipeline
logic: it exists to measure retrieval quality against a known set of
expectations, and is the one place a small amount of "we already know the
right shape of answer" bookkeeping is legitimate (see docs/superpowers/specs
for why this isn't the kind of hardcoding the project avoids everywhere
else — the ingestion/linking/retrieval code itself has no source-specific
branches).

Each query lists:
  - expected_keywords: substrings that should appear (case-insensitively) in
    good retrieved evidence — a cheap proxy for "did we get the right content".
  - expected_modalities: modalities a *good* multimodal answer should touch.
    The naive text-only baseline is structurally incapable of "visual_frame"/
    "pdf_image" modalities — that gap is the whole point of the comparison.
"""

EVAL_QUERIES = [
    {
        "id": "flagship-db-load",
        "query": "What architecture was discussed for reducing database load, who explained it, and where was the corresponding diagram shown?",
        "expected_keywords": ["memcache", "database", "load"],
        "expected_modalities": ["transcript_segment", "video_frame", "pdf_image"],
        "category": "cross_modal",
    },
    {
        "id": "mysql-load-reduction",
        "query": "How does Facebook use Memcache to reduce load on MySQL?",
        "expected_keywords": ["memcache", "mysql"],
        "expected_modalities": ["transcript_segment", "pdf_text"],
        "category": "cross_modal",
    },
    {
        "id": "look-aside-pattern",
        "query": "What problem does the look-aside cache pattern solve?",
        "expected_keywords": ["cache", "database"],
        "expected_modalities": ["transcript_segment", "pdf_text"],
        "category": "cross_modal",
    },
    {
        "id": "paper-figure",
        "query": "What does the architecture figure in the paper show?",
        "expected_keywords": ["cache", "cluster", "region", "server"],
        "expected_modalities": ["pdf_image"],
        "category": "visual",
    },
    {
        "id": "cache-invalidation",
        "query": "How is cache invalidation handled when data changes?",
        "expected_keywords": ["invalidat", "delete"],
        "expected_modalities": ["transcript_segment", "pdf_text"],
        "category": "cross_modal",
    },
    {
        "id": "thundering-herd",
        "query": "What mechanism prevents a thundering herd of requests when a cache entry is missing?",
        "expected_keywords": ["lease", "thunder"],
        "expected_modalities": ["transcript_segment", "pdf_text"],
        "category": "cross_modal",
    },
    {
        "id": "control-authors",
        "query": "Who wrote the paper on scaling Memcache at Facebook?",
        "expected_keywords": ["nishtala"],
        "expected_modalities": ["pdf_text"],
        "category": "text_control",
    },
    {
        "id": "control-what-is-memcache",
        "query": "What is Memcache?",
        "expected_keywords": ["cache", "memory"],
        "expected_modalities": ["transcript_segment", "pdf_text"],
        "category": "text_control",
    },
]
