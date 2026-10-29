"""PostgreSQL + pgvector persistence and semantic search for AutoViz."""

from autoviz.store.embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from autoviz.store.persist import build_embedding_text, persist_run_from_dir, search_charts
from autoviz.store.repository import (
    ChartHit,
    ChartRecord,
    RunRecord,
    VectorStore,
    auto_prune_dead_links,
)

__all__ = [
    "ChartHit",
    "ChartRecord",
    "Embedder",
    "HashingEmbedder",
    "RunRecord",
    "SentenceTransformerEmbedder",
    "VectorStore",
    "auto_prune_dead_links",
    "build_embedder",
    "build_embedding_text",
    "persist_run_from_dir",
    "search_charts",
]
