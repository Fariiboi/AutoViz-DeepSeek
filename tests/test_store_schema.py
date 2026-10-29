"""Tests for the pgvector schema and local embedding backends."""

from __future__ import annotations

import math

import pytest

from autoviz.store.embeddings import (
    HashingEmbedder,
    build_embedder,
    l2_normalize,
    reset_embedder_cache,
)
from autoviz.store.persist import build_embedding_text
from autoviz.store.schema import schema_statements


def test_schema_contains_vector_hnsw_and_cascades():
    ddl = "\n".join(schema_statements(512))

    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl
    assert "vector(512)" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl
    assert "ON DELETE CASCADE" in ddl
    assert "analysis_runs" in ddl and "charts" in ddl and "chart_embeddings" in ddl


@pytest.mark.parametrize("dim", [0, -1, 20000])
def test_schema_rejects_invalid_dimensions(dim):
    with pytest.raises(ValueError):
        schema_statements(dim)


def test_l2_normalize():
    vector = l2_normalize([3.0, 4.0])

    assert math.isclose(sum(value * value for value in vector), 1.0)
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(64)

    first, second = embedder.encode(["价格 与 年龄 的关系", "价格 与 年龄 的关系"])

    assert len(first) == 64
    assert first == second
    assert math.isclose(sum(value * value for value in first), 1.0)


def test_hashing_embedder_distinguishes_texts():
    embedder = HashingEmbedder(64)

    price, category = embedder.encode(["价格和年龄的关系", "各品类平均消费"])

    assert price != category


def test_build_embedder_can_use_hash_backend(monkeypatch):
    from autoviz.config import settings

    reset_embedder_cache()
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "hash")
    monkeypatch.setattr(settings, "EMBEDDING_DIM", 32)

    embedder = build_embedder()

    assert embedder.model_name == "hash-32"
    assert len(embedder.encode(["hello"])[0]) == 32
    reset_embedder_cache()


def test_embedding_text_includes_title_features_and_annotation():
    text = build_embedding_text(
        title="Price vs Age",
        chart_type="scatter",
        features=["price", "age"],
        annotation="价格随年龄上升",
    )

    assert "Price vs Age" in text
    assert "scatter" in text
    assert "price, age" in text
    assert "价格随年龄上升" in text
