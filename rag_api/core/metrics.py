"""
RAG API Custom Prometheus Metrics
===================================
Defines ML-specific Prometheus metrics that go beyond default HTTP metrics:

  - rag_query_latency_seconds   (Histogram) — end-to-end query latency
  - rag_rerank_score            (Histogram) — distribution of re-ranker scores
  - rag_documents_indexed_total (Counter)   — total documents added to the vector DB
  - rag_search_mode_total       (Counter)   — queries per search mode (vector/bm25/hybrid)
  - rag_llm_generation_errors   (Counter)   — LLM generation failures
  - rag_sources_returned        (Histogram) — number of sources returned per query

Import this module once at startup and use the metric objects directly in
your route handlers.

Usage:
    from rag_api.core.metrics import (
        RAG_QUERY_LATENCY,
        RAG_RERANK_SCORE,
        RAG_DOCUMENTS_INDEXED,
        RAG_SEARCH_MODE,
        RAG_LLM_ERRORS,
        RAG_SOURCES_RETURNED,
    )

    # In a route handler:
    with RAG_QUERY_LATENCY.labels(mode="hybrid", reranked="true").time():
        result = hybrid_search.search(...)

    RAG_SEARCH_MODE.labels(mode=body.mode).inc()
    RAG_DOCUMENTS_INDEXED.inc(len(chunks))
"""

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# Latency — tracks how long each RAG query takes end-to-end
# Labels:
#   mode     : search mode used (vector | bm25 | hybrid)
#   reranked : whether cross-encoder reranking was applied (true | false)
# ---------------------------------------------------------------------------
RAG_QUERY_LATENCY = Histogram(
    name="rag_query_latency_seconds",
    documentation="End-to-end RAG query latency in seconds",
    labelnames=["mode", "reranked"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# ---------------------------------------------------------------------------
# Re-ranker score distribution — observe the top cross-encoder score per query
# Helps detect model drift: if scores systematically drop, retrieval quality
# has degraded.
# ---------------------------------------------------------------------------
RAG_RERANK_SCORE = Histogram(
    name="rag_rerank_score",
    documentation="Distribution of top cross-encoder re-ranker scores per query",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ---------------------------------------------------------------------------
# Documents indexed — monotonically increasing counter
# Increment by number of chunks added, not raw documents.
# ---------------------------------------------------------------------------
RAG_DOCUMENTS_INDEXED = Counter(
    name="rag_documents_indexed_total",
    documentation="Total number of document chunks indexed into the vector database",
    labelnames=["source"],  # source: 'upload' | 'add_text' | 'rebuild'
)

# ---------------------------------------------------------------------------
# Search mode usage — tracks which search mode is called most
# ---------------------------------------------------------------------------
RAG_SEARCH_MODE = Counter(
    name="rag_search_mode_total",
    documentation="Total number of queries per search mode",
    labelnames=["mode"],
)

# ---------------------------------------------------------------------------
# LLM generation errors — track how often the LLM step fails
# ---------------------------------------------------------------------------
RAG_LLM_ERRORS = Counter(
    name="rag_llm_generation_errors_total",
    documentation="Total number of LLM generation failures",
    labelnames=["error_type"],
)

# ---------------------------------------------------------------------------
# Sources returned — distribution of how many sources each query returns
# ---------------------------------------------------------------------------
RAG_SOURCES_RETURNED = Histogram(
    name="rag_sources_returned",
    documentation="Number of source documents returned per query",
    buckets=[0, 1, 2, 3, 5, 7, 10, 15, 20],
)
