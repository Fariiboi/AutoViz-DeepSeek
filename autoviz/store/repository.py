"""pgvector-backed repository for runs, charts and semantic search."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from autoviz.store.schema import schema_statements

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """One analysis run persisted as a searchable unit."""

    run_id: str
    dataset_name: str
    csv_sha256: str | None = None
    row_count: int = 0
    column_count: int = 0
    sampled_rows: int = 0
    model: str = ""
    report_markdown: str = ""
    output_dir: str = ""


@dataclass
class ChartRecord:
    """One generated chart, linked to its run."""

    task_id: str
    chart_type: str
    features: list[str] = field(default_factory=list)
    title: str = ""
    annotation: str = ""
    image_path: str | None = None


@dataclass
class ChartHit:
    """A semantic-search hit."""

    chart_id: str
    title: str
    chart_type: str
    features: list[str]
    annotation: str
    image_path: str | None
    dataset_name: str
    output_dir: str
    score: float
    task_id: str = ""
    created_at: datetime | None = None


class VectorStore:
    """Thin psycopg-based repository; no ORM, explicit SQL."""

    def __init__(self, dsn: str, dim: int = 512) -> None:
        if not dsn.strip():
            raise ValueError("DATABASE_URL is empty")
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self.dsn = dsn
        self.dim = dim

    @contextmanager
    def _connect(self, *, register_vector: bool = True) -> Iterator[object]:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "psycopg 未安装。请执行：uv sync --extra rag"
            ) from exc

        with psycopg.connect(self.dsn, connect_timeout=10) as conn:
            if register_vector:
                from pgvector.psycopg import register_vector as register

                register(conn)
            yield conn

    def init_schema(self) -> None:
        """Create the pgvector extension, tables and HNSW index."""
        with self._connect(register_vector=False) as conn:
            for statement in schema_statements(self.dim):
                conn.execute(statement)

    def save_run(
        self,
        run: RunRecord,
        charts: Sequence[ChartRecord],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Upsert a run and replace its charts/embeddings atomically."""
        if len(charts) != len(vectors):
            raise ValueError("charts and vectors must have the same length")
        for vector in vectors:
            if len(vector) != self.dim:
                raise ValueError(
                    f"vector dim {len(vector)} != configured dim {self.dim}"
                )

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analysis_runs (
                        run_id, dataset_name, csv_sha256, row_count, column_count,
                        sampled_rows, model, report_markdown, output_dir
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        dataset_name = EXCLUDED.dataset_name,
                        csv_sha256 = EXCLUDED.csv_sha256,
                        row_count = EXCLUDED.row_count,
                        column_count = EXCLUDED.column_count,
                        sampled_rows = EXCLUDED.sampled_rows,
                        model = EXCLUDED.model,
                        report_markdown = EXCLUDED.report_markdown,
                        output_dir = EXCLUDED.output_dir,
                        updated_at = now()
                    """,
                    (
                        run.run_id,
                        run.dataset_name,
                        run.csv_sha256,
                        run.row_count,
                        run.column_count,
                        run.sampled_rows,
                        run.model,
                        run.report_markdown,
                        run.output_dir,
                    ),
                )

                cur.execute("DELETE FROM charts WHERE run_id = %s", (run.run_id,))

                for chart, vector in zip(charts, vectors, strict=True):
                    chart_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO charts (
                            chart_id, run_id, task_id, chart_type, features,
                            title, annotation, image_path
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chart_id,
                            run.run_id,
                            chart.task_id,
                            chart.chart_type,
                            chart.features,
                            chart.title,
                            chart.annotation,
                            chart.image_path,
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO chart_embeddings (chart_id, model, embedding)
                        VALUES (%s, %s, %s::vector)
                        """,
                        (chart_id, run.model, list(vector)),
                    )
        return len(charts)

    def search(self, query_vector: Sequence[float], limit: int = 5) -> list[ChartHit]:
        """Return the most similar charts by cosine similarity."""
        if len(query_vector) != self.dim:
            raise ValueError(
                f"query dim {len(query_vector)} != configured dim {self.dim}"
            )
        limit = max(1, min(int(limit), 50))

        from psycopg.rows import dict_row

        with self._connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        c.chart_id, c.task_id, c.title, c.chart_type, c.features,
                        c.annotation, c.image_path, c.created_at,
                        r.dataset_name, r.output_dir,
                        1 - (e.embedding <=> %s::vector) AS score
                    FROM chart_embeddings e
                    JOIN charts c ON c.chart_id = e.chart_id
                    JOIN analysis_runs r ON r.run_id = c.run_id
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (list(query_vector), list(query_vector), limit),
                )
                rows = cur.fetchall()

        return [
            ChartHit(
                chart_id=str(row["chart_id"]),
                title=row["title"] or "",
                chart_type=row["chart_type"] or "",
                features=list(row["features"] or []),
                annotation=row["annotation"] or "",
                image_path=row["image_path"],
                dataset_name=row["dataset_name"] or "",
                output_dir=row["output_dir"] or "",
                score=float(row["score"]),
                task_id=row["task_id"] or "",
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def stats(self) -> dict[str, int]:
        """Return row counts for diagnostics."""
        from psycopg.rows import dict_row

        with self._connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM analysis_runs) AS runs,
                        (SELECT count(*) FROM charts) AS charts,
                        (SELECT count(*) FROM chart_embeddings) AS embeddings
                    """
                )
                row = cur.fetchone()
        return {key: int(value) for key, value in (row or {}).items()}

    def delete_run(self, run_id: str) -> None:
        """Delete a run and (via cascade) its charts and embeddings."""
        with self._connect() as conn:
            conn.execute("DELETE FROM analysis_runs WHERE run_id = %s", (run_id,))

    def prune_dead_artifacts(self) -> dict[str, int]:
        """Delete runs whose output_dir is gone and charts whose image is missing.

        Returns the number of removed runs/charts (embeddings cascade with them).
        """
        removed_runs = 0
        removed_charts = 0

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT run_id, output_dir FROM analysis_runs")
                for run_id, output_dir in cur.fetchall():
                    if not output_dir or not Path(output_dir).is_dir():
                        cur.execute("DELETE FROM analysis_runs WHERE run_id = %s", (run_id,))
                        removed_runs += 1

                cur.execute("SELECT chart_id, image_path FROM charts")
                for chart_id, image_path in cur.fetchall():
                    if not image_path:
                        continue
                    path = Path(image_path)
                    if not path.is_absolute():
                        path = Path.cwd() / path
                    if not path.exists():
                        cur.execute("DELETE FROM charts WHERE chart_id = %s", (chart_id,))
                        removed_charts += 1

        return {"runs": removed_runs, "charts": removed_charts}


def auto_prune_dead_links(
    *,
    store: VectorStore | None = None,
    dsn: str | None = None,
    dim: int | None = None,
) -> dict[str, int | str]:
    """Best-effort dead-link cleanup that never raises.

    Called silently by the UI (page load / before search). A failure is logged
    and returned in the ``error`` key instead of breaking the page.
    """
    try:
        if store is None:
            from autoviz.config import settings

            dsn = (dsn or settings.DATABASE_URL).strip()
            if not dsn:
                return {"runs": 0, "charts": 0}
            store = VectorStore(dsn, int(dim or settings.EMBEDDING_DIM))

        result = store.prune_dead_artifacts()
        if result.get("runs") or result.get("charts"):
            logger.info("auto-pruned dead links: %s", result)
        return result
    except Exception as exc:  # cleanup must never break callers
        logger.warning("auto prune failed (ignored): %s", exc)
        return {"runs": 0, "charts": 0, "error": str(exc)}
