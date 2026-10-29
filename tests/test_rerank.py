"""Tests for CrossEncoder reranking helpers (no model download)."""

from __future__ import annotations

import pytest

from autoviz.store.repository import ChartHit
from autoviz.store.rerank import (
    candidate_text,
    relevance_probability,
    rerank_hits,
)


def _hit(title: str, annotation: str = "", score: float = 0.5) -> ChartHit:
    return ChartHit(
        chart_id=title,
        title=title,
        chart_type="scatter",
        features=["age", "amount"],
        annotation=annotation,
        image_path=None,
        dataset_name="order.csv",
        output_dir="D:/autoviz-output/order/run",
        score=score,
    )


class FakeReranker:
    model_name = "fake-reranker"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [10.0 if "年龄" in text else 1.0 for text in texts]


def test_candidate_text_includes_title_features_and_annotation():
    text = candidate_text(_hit("年龄与消费", "年龄越大消费越高"))

    assert "年龄与消费" in text
    assert "age, amount" in text
    assert "年龄越大消费越高" in text


def test_rerank_orders_by_cross_score_and_truncates():
    hits = [
        _hit("时段热力图", "", score=0.9),
        _hit("年龄与消费", "年龄与消费正相关", score=0.4),
        _hit("地区分布", "", score=0.8),
    ]

    ranked = rerank_hits("年龄", hits, FakeReranker(), top_k=2)

    assert len(ranked) == 2
    assert ranked[0][0].title == "年龄与消费"
    assert ranked[0][1] == 10.0


def test_rerank_rejects_mismatched_score_count():
    class BrokenReranker:
        model_name = "broken"

        def score(self, query: str, texts: list[str]) -> list[float]:
            return [1.0]

    with pytest.raises(ValueError):
        rerank_hits("年龄", [_hit("a"), _hit("b")], BrokenReranker(), top_k=2)


def test_relevance_probability_keeps_sigmoid_scores_and_maps_logits():
    assert relevance_probability(0.0) == pytest.approx(0.0)
    assert relevance_probability(0.93) == pytest.approx(0.93)
    assert 0.0 < relevance_probability(-10.0) < 0.01
    assert 0.99 < relevance_probability(10.0) <= 1.0
