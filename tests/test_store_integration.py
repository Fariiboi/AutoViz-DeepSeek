"""Integration test for the pgvector store.

Runs only when ``AUTOVIZ_TEST_DATABASE_URL`` points at a PostgreSQL instance
with the pgvector extension installed (for example the docker-compose service).
"""

from __future__ import annotations

import os
import uuid

import pytest

from autoviz.store.embeddings import HashingEmbedder
from autoviz.store.persist import build_embedding_text
from autoviz.store.repository import ChartRecord, RunRecord, VectorStore

DSN = os.getenv("AUTOVIZ_TEST_DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="AUTOVIZ_TEST_DATABASE_URL is not configured"
)


def test_store_roundtrip_and_semantic_search():
    dim = 512
    store = VectorStore(DSN, dim)
    store.init_schema()

    embedder = HashingEmbedder(dim)
    run_id = str(uuid.uuid4())
    charts = [
        ChartRecord(
            task_id="scatter-1",
            chart_type="scatter",
            features=["price", "age"],
            title="Price vs Age",
            annotation="价格与年龄的散点关系, 中青年群体消费更高。",
        ),
        ChartRecord(
            task_id="bar-2",
            chart_type="bar",
            features=["category_code", "price"],
            title="Average Price by Category",
            annotation="各商品品类的平均价格对比。",
        ),
    ]
    vectors = embedder.encode(
        [
            build_embedding_text(
                title=chart.title,
                chart_type=chart.chart_type,
                features=chart.features,
                annotation=chart.annotation,
            )
            for chart in charts
        ]
    )
    run = RunRecord(
        run_id=run_id,
        dataset_name="test.csv",
        row_count=100,
        column_count=3,
        model=embedder.model_name,
    )

    try:
        assert store.save_run(run, charts, vectors) == 2

        hits = store.search(embedder.encode(["价格和年龄的关系"])[0], limit=2)

        assert hits
        assert hits[0].title == "Price vs Age"
        assert hits[0].dataset_name == "test.csv"
        assert 0.0 <= hits[0].score <= 1.0

        stats = store.stats()
        assert stats["runs"] >= 1
        assert stats["charts"] >= 2
    finally:
        store.delete_run(run_id)


def test_prune_removes_missing_runs_and_images(tmp_path):
    dim = 512
    store = VectorStore(DSN, dim)
    store.init_schema()
    embedder = HashingEmbedder(dim)

    ghost_run_id = str(uuid.uuid4())
    live_run_id = str(uuid.uuid4())

    live_chart = ChartRecord(
        task_id="missing-image",
        chart_type="bar",
        title="Chart with deleted PNG",
        image_path=str(tmp_path / "not-there.png"),
    )

    try:
        store.save_run(
            RunRecord(run_id=ghost_run_id, dataset_name="ghost", output_dir=str(tmp_path / "gone")),
            [],
            [],
        )
        store.save_run(
            RunRecord(run_id=live_run_id, dataset_name="live", output_dir=str(tmp_path)),
            [live_chart],
            embedder.encode(["chart with deleted png"]),
        )

        removed = store.prune_dead_artifacts()

        assert removed["runs"] >= 1  # ghost run removed
        assert removed["charts"] >= 1  # chart whose PNG is gone removed
    finally:
        store.delete_run(ghost_run_id)
        store.delete_run(live_run_id)

