"""Persist an AutoViz run (charts + annotations) into pgvector."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from autoviz.store.embeddings import Embedder
from autoviz.store.repository import ChartHit, ChartRecord, RunRecord, VectorStore

logger = logging.getLogger(__name__)


def build_embedding_text(
    *,
    title: str,
    chart_type: str,
    features: list[str],
    annotation: str,
) -> str:
    """Compose the text that represents a chart in vector space.

    Annotations carry most of the semantic signal; title/type/features anchor
    short or missing annotations so the record is still retrievable.
    """
    parts = [
        f"图表标题：{title}",
        f"图表类型：{chart_type}",
        f"字段：{', '.join(features)}",
    ]
    if annotation.strip():
        parts.append(f"洞察：{annotation.strip()}")
    return "\n".join(parts)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cannot read %s: %s", path, exc)
        return default


def _history_image_map(run_dir: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """Map (chart_type, features_key) -> (task_id, image_path) from history.md.

    Older runs were produced before ``charts_metadata.json`` stored task_id /
    image_path, so history.md is the only place that links metadata to PNGs.
    """
    history_path = run_dir / "history.md"
    if not history_path.exists():
        return {}

    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("## ["):
            continue
        parts = [part.strip() for part in line[3:].split("|")]
        if len(parts) < 6:
            continue
        task_id, chart_type, features_key, status, path = parts[1:6]
        if status not in {"done", "in_progress"} or path in {"", "-"}:
            continue
        mapping[(chart_type, features_key)] = (task_id, path)
    return mapping


def _absolute_image_path(raw_path: str, run_dir: Path) -> str:
    """Resolve an image path to an absolute path (never store a bare relative path)."""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path.resolve())

    candidates = [Path.cwd() / path]
    if len(run_dir.parents) >= 3:
        candidates.append(run_dir.parents[2] / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def persist_run_from_dir(
    run_dir: Path,
    *,
    dataset_name: str,
    store: VectorStore,
    embedder: Embedder,
    csv_sha256: str | None = None,
) -> int:
    """Read ``csv_summary.json`` / ``charts_metadata.json`` / ``report.md`` and upsert.

    The run id is derived from the absolute output directory, so re-ingesting
    the same directory replaces the previous row set instead of duplicating it.
    """
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    summary = _load_json(run_dir / "csv_summary.json", {}) or {}
    metadata = _load_json(run_dir / "charts_metadata.json", []) or []
    report_path = run_dir / "report.md"
    report_markdown = (
        report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    )

    charts: list[ChartRecord] = []
    texts: list[str] = []
    history_images = _history_image_map(run_dir)
    for index, item in enumerate(metadata):
        if not isinstance(item, dict):
            continue
        features = [str(feature) for feature in (item.get("features") or [])]
        chart_type = str(item.get("chart_type") or "unknown")
        image_path = item.get("image_path")
        task_id = item.get("task_id")

        if not image_path:
            features_key = "+".join(sorted(features))
            fallback = history_images.get((chart_type, features_key))
            if fallback:
                task_id = task_id or fallback[0]
                image_path = fallback[1]

        if image_path:
            image_path = _absolute_image_path(str(image_path), run_dir)

        task_id = str(task_id or (Path(image_path).stem if image_path else f"chart-{index + 1}"))
        chart = ChartRecord(
            task_id=task_id,
            chart_type=chart_type,
            features=features,
            title=str(item.get("title") or ""),
            annotation=str(item.get("annotation") or ""),
            image_path=str(image_path) if image_path else None,
        )
        charts.append(chart)
        texts.append(
            build_embedding_text(
                title=chart.title,
                chart_type=chart.chart_type,
                features=chart.features,
                annotation=chart.annotation,
            )
        )

    if not charts:
        logger.info("no charts in %s; nothing to persist", run_dir)
        return 0

    vectors = embedder.encode(texts)
    run = RunRecord(
        run_id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(run_dir))),
        dataset_name=dataset_name,
        csv_sha256=csv_sha256,
        row_count=int(summary.get("row_count") or 0),
        column_count=int(summary.get("column_count") or 0),
        sampled_rows=int(summary.get("sampled_rows") or 0),
        model=embedder.model_name,
        report_markdown=report_markdown,
        output_dir=str(run_dir),
    )
    return store.save_run(run, charts, vectors)


def search_charts(
    store: VectorStore,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 5,
) -> list[ChartHit]:
    """Embed ``query`` and return the closest charts."""
    query = query.strip()
    if not query:
        return []
    vector = embedder.encode([query])[0]
    return store.search(vector, limit=limit)
