"""AutoViz CLI — entry point and analyze pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import click

from autoviz.agents.reporter import DataReport, VizProperty, generate_report
from autoviz.agents.translator import translate_report_to_chinese
from autoviz.config import settings
from autoviz.events import VizResult
from autoviz.pipeline.charts import run_visualization
from autoviz.pipeline.csv import CsvSummary, preprocess_csv_with_data, read_text_auto
from autoviz.pipeline.features import run_feature_engineering

logger = logging.getLogger(__name__)

_METADATA_FILE = "charts_metadata.json"
_SUMMARY_FILE = "csv_summary.json"


# ── Report helpers ────────────────────────────────────────────────────────────

def _build_viz_properties(results: list[VizResult]) -> list[VizProperty]:
    return [
        VizProperty(
            title=(result.metadata or {}).get("title", result.chart_type),
            chart_type=result.chart_type,
            features=list(result.features),
            annotation=(result.metadata or {}).get("annotation"),
        )
        for result in results
        if result.status == "done"
    ]


def _report_to_markdown(report: DataReport) -> str:
    lines = ["# AutoViz Analysis Report", "", "## Dataset Overview", "", report.dataset_overview, ""]

    lines += ["## Key Findings", ""]
    for theme in report.key_findings:
        lines.append(f"### {theme.theme}")
        lines.append(theme.summary)
        for detail in theme.details:
            lines.append(f"- {detail.text}")
        lines.append("")

    if report.relationship_analysis:
        lines += ["## Relationship Analysis", ""]
        for insight in report.relationship_analysis:
            lines.append(f"**{', '.join(insight.variables)}**: {insight.description}")
            lines.append("")

    if report.suggestions:
        lines += ["## Suggestions", ""]
        for s in report.suggestions:
            lines.append(f"- {s}")

    return "\n".join(lines)


async def _save_report(
    csv_summary: CsvSummary,
    viz_props: list[VizProperty],
    output_dir: Path,
    fmt: str,
) -> None:
    click.echo("[AutoViz] Generating report...")
    try:
        report = await generate_report(csv_summary, viz_props)
        english_markdown = _report_to_markdown(report)
        if fmt == "json":
            report_path = output_dir / "report.json"
            report_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
        else:
            report_path = output_dir / "report.md"
            report_path.write_text(english_markdown, encoding="utf-8")
        click.secho(f"[AutoViz] Report saved to {report_path}", fg="green")

        # Bilingual output: translate the English Markdown into a Chinese business report.
        try:
            chinese_markdown = await translate_report_to_chinese(english_markdown)
            chinese_path = output_dir / "report_chn.md"
            chinese_path.write_text(chinese_markdown, encoding="utf-8")
            click.secho(f"[AutoViz] Chinese report saved to {chinese_path}", fg="green")
        except Exception as exc:  # translation must not fail the run
            logger.warning("Chinese report generation failed: %s", exc)
            click.secho(
                f"[AutoViz] Chinese report generation failed: {exc}",
                fg="yellow",
            )
    except Exception as e:
        click.secho(f"[AutoViz] Report generation failed: {e}", fg="red")
        logger.exception("Report generation error")


# ── Pipeline functions ────────────────────────────────────────────────────────

async def run_analyze_pipeline(
    csv_path: Path,
    output_dir: Path,
    target_charts: int,
    no_feature_eng: bool,
    with_report: bool,
    report_format: str,
) -> None:
    """Full pipeline: CSV → preprocess → feature eng → visualize → (report)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse CSV (UTF-8 first, GB18030/GBK fallback)
    click.echo(f"[AutoViz] Parsing CSV: {csv_path.name}")
    try:
        csv_content = read_text_auto(csv_path)
    except Exception as e:
        click.secho(f"[AutoViz] Cannot read file: {e}", fg="red")
        return

    csv_sha256 = hashlib.sha256(csv_content.encode("utf-8")).hexdigest()

    csv_summary, preprocessed_csv = preprocess_csv_with_data(csv_content)
    if csv_summary.error:
        click.secho(f"[AutoViz] CSV parse failed: {csv_summary.error}", fg="red")
        return

    if csv_summary.sampled_rows:
        click.echo(
            f"[AutoViz] 检测到 {csv_summary.row_count_before_sampling:,} 行数据，"
            f"已自动采样 {csv_summary.sampled_rows:,} 行进行分析。"
        )

    click.echo(
        f"[AutoViz] CSV parsed: {csv_summary.row_count:,} rows, {csv_summary.column_count} columns"
    )
    (output_dir / _SUMMARY_FILE).write_text(
        json.dumps(csv_summary.to_dict(), indent=2), encoding="utf-8"
    )

    # 2. Feature engineering
    enhanced_csv = preprocessed_csv
    if not no_feature_eng:
        click.echo("[AutoViz] Feature engineering...")
        try:
            enhanced_csv = await run_feature_engineering(
                csv_summary,
                preprocessed_csv,
                on_stdout=None,
                on_stderr=None,
            )
            enhanced_summary, _ = preprocess_csv_with_data(enhanced_csv)
            if not enhanced_summary.error:
                csv_summary = enhanced_summary
                (output_dir / _SUMMARY_FILE).write_text(
                    json.dumps(csv_summary.to_dict(), indent=2), encoding="utf-8"
                )
            click.echo("[AutoViz] Feature engineering done")
        except Exception as e:
            click.secho(
                f"[AutoViz] Feature engineering failed ({e}), continuing with original CSV",
                fg="yellow",
            )
            enhanced_csv = preprocessed_csv

    # 3. Visualization
    click.echo(f"[AutoViz] Generating up to {target_charts} chart(s)...")

    done_count = 0
    skip_count = 0

    def on_result(result: VizResult) -> None:
        nonlocal done_count, skip_count
        md = result.metadata or {}
        title = md.get("title", result.chart_type)
        feat = ", ".join(result.features)
        if result.status == "done" and result.image_path:
            done_count += 1
            click.echo(f"  + {title} [{feat}] -> {result.image_path}")
        elif result.status in ("skipped", "error"):
            skip_count += 1
            reason = md.get("critic_reason") or md.get("skip_kind") or result.error or ""
            click.echo(f"  - {title} [{feat}] -> {result.status} ({reason})")

    results = await run_visualization(
        enhanced_csv=enhanced_csv,
        csv_summary=csv_summary,
        on_event=on_result,
        target_charts=target_charts,
        num_turns=1,
        viz_workspace_override=charts_dir,
    )

    click.secho(
        f"[AutoViz] Done: {done_count} chart(s) saved to {charts_dir}, {skip_count} skipped",
        fg="green",
    )

    # 4. Persist chart metadata
    viz_props = _build_viz_properties(results)
    metadata_items = [
        {
            "task_id": result.task_id,
            "title": (result.metadata or {}).get("title", result.chart_type),
            "chart_type": result.chart_type,
            "features": list(result.features),
            "annotation": (result.metadata or {}).get("annotation"),
            "image_path": result.image_path,
        }
        for result in results
        if result.status == "done"
    ]
    (output_dir / _METADATA_FILE).write_text(
        json.dumps(metadata_items, indent=2),
        encoding="utf-8",
    )

    # 5. Optional report
    if with_report:
        if viz_props:
            await _save_report(csv_summary, viz_props, output_dir, report_format)
        else:
            click.secho("[AutoViz] No accepted charts; skipping report.", fg="yellow")

    # 6. Optional pgvector persistence (fail-open: never breaks the analysis)
    _persist_run_if_configured(output_dir, csv_path, csv_sha256=csv_sha256)


def _persist_run_if_configured(
    run_dir: Path,
    csv_path: Path,
    *,
    csv_sha256: str | None = None,
) -> None:
    """Write the finished run into pgvector when DATABASE_URL is configured."""
    if not settings.DATABASE_URL.strip():
        return
    try:
        from autoviz.store.embeddings import build_embedder
        from autoviz.store.persist import persist_run_from_dir
        from autoviz.store.repository import VectorStore

        store = VectorStore(settings.DATABASE_URL, settings.EMBEDDING_DIM)
        count = persist_run_from_dir(
            run_dir,
            dataset_name=csv_path.name,
            store=store,
            embedder=build_embedder(),
            csv_sha256=csv_sha256,
        )
        if count:
            click.secho(f"[AutoViz] 向量库已写入 {count} 张图表", fg="green")
    except Exception as exc:
        logger.warning("vector store persistence failed: %s", exc)
        click.secho(f"[AutoViz] 向量库写入失败（不影响本次分析）：{exc}", fg="yellow")


async def run_report_only(output_dir: Path, fmt: str) -> None:
    """Generate a report from a previous analyze run's output directory."""
    summary_path = output_dir / _SUMMARY_FILE
    metadata_path = output_dir / _METADATA_FILE

    if not summary_path.exists():
        click.secho(f"[AutoViz] {_SUMMARY_FILE} not found in {output_dir}", fg="red")
        click.echo("Run 'autoviz analyze run <file.csv>' first.")
        return
    if not metadata_path.exists():
        click.secho(f"[AutoViz] {_METADATA_FILE} not found in {output_dir}", fg="red")
        click.echo("Run 'autoviz analyze run <file.csv>' first.")
        return

    csv_summary = CsvSummary.from_dict(json.loads(summary_path.read_text(encoding="utf-8")))
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    viz_props = [
        VizProperty(
            title=m["title"],
            chart_type=m["chart_type"],
            features=m["features"],
            annotation=m.get("annotation"),
        )
        for m in raw
    ]

    if not viz_props:
        click.secho("[AutoViz] No charts found in metadata; nothing to report.", fg="yellow")
        return

    await _save_report(csv_summary, viz_props, output_dir, fmt)


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="autoviz")
def cli():
    """AutoViz — AI-powered CSV visualization pipeline."""


@cli.group("analyze")
def analyze_cli():
    """Analyze a CSV file and generate visualizations + reports."""


@analyze_cli.command("run")
@click.argument("csv_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    default="./autoviz-output",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Base output directory. Results go to OUTPUT/<csv-name>/<timestamp>/",
)
@click.option("--charts", "-n", default=10, show_default=True, type=int,
              help="Number of charts to generate")
@click.option("--no-feature-eng", is_flag=True,
              help="Skip feature engineering (faster, uses raw CSV columns)")
@click.option("--report", "with_report", is_flag=True,
              help="Generate a written report after charts are done")
@click.option(
    "--format", "fmt",
    default="md",
    show_default=True,
    type=click.Choice(["md", "json"]),
    help="Report format",
)
def analyze_run(
    csv_file: Path,
    output: Path,
    charts: int,
    no_feature_eng: bool,
    with_report: bool,
    fmt: str,
) -> None:
    """Analyze CSV_FILE and generate AI-powered visualizations.

    Charts are saved as PNG files under OUTPUT/<csv-name>/<timestamp>/charts/.
    Requires DEEPSEEK_API_KEY and MODAL_TOKEN_ID/SECRET in .env.

    \b
    Examples:
      autoviz analyze run sales.csv
      autoviz analyze run sales.csv -o ./results -n 5 --report
      autoviz analyze run sales.csv --no-feature-eng --report --format json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output / csv_file.stem / timestamp
    click.echo(f"[AutoViz] Output directory: {run_dir}")
    asyncio.run(run_analyze_pipeline(csv_file, run_dir, charts, no_feature_eng, with_report, fmt))


@analyze_cli.command("report")
@click.option(
    "--from", "source_dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a previous analyze run (e.g. autoviz-output/sales/20260419_143022)",
)
@click.option(
    "--format", "fmt",
    default="md",
    show_default=True,
    type=click.Choice(["md", "json"]),
    help="Report format",
)
def analyze_report(source_dir: Path, fmt: str) -> None:
    """Generate a report from charts produced by a previous 'analyze run'.

    \b
    Examples:
      autoviz analyze report --from autoviz-output/sales/20260419_143022
      autoviz analyze report --from ./results/sales/20260419_143022 --format json
    """
    asyncio.run(run_report_only(source_dir, fmt))


def _database_url_or_fail() -> str:
    url = settings.DATABASE_URL.strip()
    if not url:
        raise click.ClickException(
            "未配置 DATABASE_URL。请参考 .env.example 启用 PostgreSQL + pgvector。"
        )
    return url


@cli.group("db")
def db_cli():
    """PostgreSQL + pgvector 向量库管理。"""


@db_cli.command("init")
def db_init() -> None:
    """创建 pgvector 扩展、表结构与 HNSW 索引。"""
    from autoviz.store.repository import VectorStore

    try:
        store = VectorStore(_database_url_or_fail(), settings.EMBEDDING_DIM)
        store.init_schema()
        stats = store.stats()
    except Exception as exc:
        raise click.ClickException(f"初始化失败：{exc}") from exc
    click.secho("[AutoViz] pgvector schema ready", fg="green")
    click.echo(f"  runs={stats.get('runs', 0)} charts={stats.get('charts', 0)}")


@db_cli.command("ingest")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def db_ingest(run_dir: Path) -> None:
    """把已有运行目录写入向量库（用于回填历史结果）。"""
    from autoviz.store.embeddings import build_embedder
    from autoviz.store.persist import persist_run_from_dir
    from autoviz.store.repository import VectorStore

    try:
        store = VectorStore(_database_url_or_fail(), settings.EMBEDDING_DIM)
        store.init_schema()
        count = persist_run_from_dir(
            run_dir,
            dataset_name=run_dir.parent.name,
            store=store,
            embedder=build_embedder(),
        )
    except Exception as exc:
        raise click.ClickException(f"写入失败：{exc}") from exc
    click.secho(f"[AutoViz] 已写入 {count} 张图表（{run_dir}）", fg="green")


@db_cli.command("stats")
def db_stats() -> None:
    """查看 runs / charts / embeddings 数量。"""
    from autoviz.store.repository import VectorStore

    try:
        store = VectorStore(_database_url_or_fail(), settings.EMBEDDING_DIM)
        stats = store.stats()
    except Exception as exc:
        raise click.ClickException(f"查询失败：{exc}") from exc
    click.echo(
        f"runs={stats.get('runs', 0)} "
        f"charts={stats.get('charts', 0)} "
        f"embeddings={stats.get('embeddings', 0)}"
    )


@db_cli.command("prune")
def db_prune() -> None:
    """清理死链：删除目录已不存在或图片文件缺失的向量库记录。"""
    from autoviz.store.repository import VectorStore

    try:
        store = VectorStore(_database_url_or_fail(), settings.EMBEDDING_DIM)
        removed = store.prune_dead_artifacts()
        stats = store.stats()
    except Exception as exc:
        raise click.ClickException(f"清理失败：{exc}") from exc
    click.secho(
        f"[AutoViz] 已清理 runs={removed['runs']} charts={removed['charts']}",
        fg="green",
    )
    click.echo(
        f"剩余 runs={stats.get('runs', 0)} "
        f"charts={stats.get('charts', 0)} "
        f"embeddings={stats.get('embeddings', 0)}"
    )


@cli.command("search")
@click.argument("query")
@click.option("-k", "--limit", type=int, default=None, help="返回条数（默认 RAG_TOP_K）")
def search_cmd(query: str, limit: int | None) -> None:
    """语义检索历史图表（需要 DATABASE_URL + 本地 embedding 模型）。"""
    from autoviz.store.embeddings import build_embedder
    from autoviz.store.persist import search_charts
    from autoviz.store.repository import VectorStore

    top_k = limit if limit is not None else settings.RAG_TOP_K
    try:
        store = VectorStore(_database_url_or_fail(), settings.EMBEDDING_DIM)
        hits = search_charts(store, build_embedder(), query, limit=top_k)
    except Exception as exc:
        raise click.ClickException(f"检索失败：{exc}") from exc

    if not hits:
        click.echo("没有匹配结果；请先运行 `autoviz analyze run ...` 或 `autoviz db ingest ...`。")
        return

    click.secho(f"[AutoViz] Top {len(hits)} results for: {query}", fg="green")
    for index, hit in enumerate(hits, start=1):
        click.echo(f"\n{index}. [{hit.score:.3f}] {hit.title}  ({hit.chart_type})")
        click.echo(f"   dataset: {hit.dataset_name} | features: {', '.join(hit.features)}")
        if hit.image_path:
            click.echo(f"   image:   {hit.image_path}")
        annotation = " ".join(hit.annotation.split())
        if annotation:
            click.echo(f"   note:    {annotation[:180]}")


def main():
    cli()


if __name__ == "__main__":
    main()
