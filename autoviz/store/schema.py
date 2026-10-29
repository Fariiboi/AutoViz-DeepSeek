"""PostgreSQL + pgvector schema for AutoViz semantic search."""

from __future__ import annotations

SCHEMA_VERSION = 1


def schema_statements(dim: int = 512) -> list[str]:
    """Return idempotent DDL statements for the vector store.

    Args:
        dim: Embedding dimension; must match the embedder and ``vector(dim)``.
    """
    if dim <= 0 or dim > 16000:
        raise ValueError(f"invalid embedding dimension: {dim}")

    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
            run_id          UUID PRIMARY KEY,
            dataset_name    TEXT NOT NULL,
            csv_sha256      TEXT,
            row_count       INTEGER NOT NULL DEFAULT 0,
            column_count    INTEGER NOT NULL DEFAULT 0,
            sampled_rows    INTEGER NOT NULL DEFAULT 0,
            model           TEXT NOT NULL DEFAULT '',
            report_markdown TEXT NOT NULL DEFAULT '',
            output_dir      TEXT NOT NULL DEFAULT '',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS charts (
            chart_id    UUID PRIMARY KEY,
            run_id      UUID NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
            task_id     TEXT NOT NULL DEFAULT '',
            chart_type  TEXT NOT NULL,
            features    TEXT[] NOT NULL DEFAULT '{}',
            title       TEXT NOT NULL DEFAULT '',
            annotation  TEXT NOT NULL DEFAULT '',
            image_path  TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        "CREATE INDEX IF NOT EXISTS charts_run_id_idx ON charts(run_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS charts_run_task_uidx ON charts(run_id, task_id)",
        f"""
        CREATE TABLE IF NOT EXISTS chart_embeddings (
            chart_id   UUID PRIMARY KEY REFERENCES charts(chart_id) ON DELETE CASCADE,
            model      TEXT NOT NULL,
            embedding  vector({dim}) NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS chart_embeddings_hnsw_idx "
        "ON chart_embeddings USING hnsw (embedding vector_cosine_ops)",
    ]
