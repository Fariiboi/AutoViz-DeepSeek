"""CrossEncoder reranking for semantic chart search (retrieve & rerank)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Protocol

from autoviz.config import settings
from autoviz.store.embeddings import model_is_cached
from autoviz.store.repository import ChartHit

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Minimal reranker interface (cross-encoder or test double)."""

    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one relevance score per candidate text."""


def candidate_text(hit: ChartHit, max_annotation_chars: int = 2_000) -> str:
    """Text representation of a chart used for query-document cross scoring."""
    parts = [
        f"标题：{hit.title}",
        f"图表类型：{hit.chart_type}",
        f"字段：{', '.join(hit.features)}",
    ]
    annotation = (hit.annotation or "").strip()
    if annotation:
        parts.append(f"标注：{annotation[:max_annotation_chars]}")
    return "\n".join(parts)


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder reranker (BGE-reranker by default)."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._batch_size = batch_size
        cached = model_is_cached(model_name)
        try:
            self._model = CrossEncoder(
                model_name,
                device=device,
                max_length=max_length,
                local_files_only=cached,
            )
        except Exception:
            # Repair a partially cached snapshot from the configured mirror.
            if not cached:
                raise
            logger.warning("cached reranker incomplete; retrying download")
            self._model = CrossEncoder(
                model_name,
                device=device,
                max_length=max_length,
                local_files_only=False,
            )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        raw = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        try:
            return [float(value) for value in raw]
        except TypeError:
            return [float(raw)]


def build_reranker(
    model_name: str | None = None,
    device: str | None = None,
) -> CrossEncoderReranker:
    """Build the configured CrossEncoder reranker."""
    return CrossEncoderReranker(
        model_name or settings.RERANK_MODEL,
        device=device or settings.RERANK_DEVICE,
        max_length=settings.RERANK_MAX_LENGTH,
        batch_size=settings.RERANK_BATCH_SIZE,
    )


def rerank_hits(
    query: str,
    hits: Sequence[ChartHit],
    reranker: Reranker,
    *,
    top_k: int,
) -> list[tuple[ChartHit, float]]:
    """Cross-score query/candidate pairs and return the best ``top_k``."""
    if not hits:
        return []
    texts = [candidate_text(hit) for hit in hits]
    scores = reranker.score(query, texts)
    if len(scores) != len(hits):
        raise ValueError(
            f"reranker returned {len(scores)} scores for {len(hits)} candidates"
        )
    ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return ranked[: max(1, int(top_k))]


def relevance_probability(score: float) -> float:
    """Return the display relevance; BGE rerankers already apply sigmoid."""
    value = float(score)
    if 0.0 <= value <= 1.0:
        return value
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0
