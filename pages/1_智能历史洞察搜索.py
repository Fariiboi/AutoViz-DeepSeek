"""Streamlit page: executive summaries per analysis batch + semantic chart search."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import streamlit as st

from autoviz.agents.summarizer import generate_executive_summary
from autoviz.config import settings
from autoviz.store.rerank import relevance_probability, rerank_hits

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "autoviz-output"

st.set_page_config(page_title="智能历史洞察搜索 · AutoViz", page_icon="🧠", layout="wide")


def _auto_prune(*, force: bool = False) -> int:
    """Silently remove dead links once per session (and before each search)."""
    if not settings.DATABASE_URL.strip():
        return 0
    if not force and st.session_state.get("auto_prune_checked"):
        return 0

    from autoviz.store.repository import auto_prune_dead_links

    result = auto_prune_dead_links()
    st.session_state["auto_prune_checked"] = True
    removed = int(result.get("runs") or 0) + int(result.get("charts") or 0)
    if removed:
        st.session_state["auto_prune_removed"] = removed
    return removed


st.title("🧠 智能历史洞察搜索")
st.caption(
    "选择一次历史分析，让 DeepSeek 为老板写一份商业洞察总结；"
    "也可以用自然语言搜索历史图表。"
)

_auto_prune()
if st.session_state.get("auto_prune_removed"):
    st.caption(f"🧹 已自动清理 {st.session_state['auto_prune_removed']} 条失效历史记录")


def _scan_run_dirs() -> list[Path]:
    """Return completed run directories under autoviz-output, newest first."""
    if not OUTPUT_ROOT.exists():
        return []
    run_dirs = [
        path
        for path in OUTPUT_ROOT.glob("*/*")
        if path.is_dir() and (path / "report.md").exists()
    ]
    return sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def _load_run_material(run_dir: Path) -> tuple[str, list[dict], dict]:
    """Load report.md, charts_metadata.json (or PNG fallback) and csv_summary.json."""
    report_path = run_dir / "report.md"
    if not report_path.exists():
        raise FileNotFoundError(f"未找到报告文件：{report_path}")
    report_markdown = report_path.read_text(encoding="utf-8")

    charts: list[dict] = []
    metadata_path = run_dir / "charts_metadata.json"
    if metadata_path.exists():
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = [raw]
        charts = [item for item in raw if isinstance(item, dict)]

    if not charts:
        charts_dir = run_dir / "charts"
        pngs = sorted(charts_dir.glob("*.png")) if charts_dir.is_dir() else []
        charts = [
            {
                "title": png.stem,
                "chart_type": "chart",
                "features": [],
                "annotation": "",
            }
            for png in pngs
        ]

    dataset: dict = {}
    summary_path = run_dir / "csv_summary.json"
    if summary_path.exists():
        try:
            dataset = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("cannot parse %s", summary_path)
            dataset = {}

    return report_markdown, charts, dataset


@st.cache_resource(show_spinner=False)
def _embedder():
    from autoviz.store.embeddings import build_embedder

    return build_embedder()


@st.cache_resource(show_spinner=False)
def _reranker():
    from autoviz.store.rerank import build_reranker

    return build_reranker()


def _store():
    from autoviz.store.repository import VectorStore

    return VectorStore(settings.DATABASE_URL, settings.EMBEDDING_DIM)


def _resolve_image_path(hit) -> tuple[Path | None, str]:
    """Resolve DB paths (relative or absolute) and reconstruct a fallback path.

    Returns (existing_path_or_None, path_for_debug). Prints the resolved path to
    the Streamlit server log so path issues are visible in the terminal.
    """
    candidates: list[Path] = []
    if hit.image_path:
        candidates.append(Path(hit.image_path))
    if hit.output_dir and hit.task_id:
        candidates.append(Path(hit.output_dir) / "charts" / f"{hit.task_id}.png")

    debug_path = "（数据库未记录图片路径）"
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        path = Path(os.path.normpath(str(path)))
        try:
            path = path.resolve()
        except OSError:
            pass
        debug_path = str(path)
        print(f"尝试读取的图片路径: {path} (exists={path.exists()})")
        if path.exists():
            return path, debug_path
    return None, debug_path


# ── 1. 按批次生成商业洞察总结 ────────────────────────────────────────────────

st.subheader("📁 按分析批次生成汇总报告")
st.caption("把该批次的 report.md 与全部图表标注拼成一份材料，交给 DeepSeek 写成面向老板的总结。")

run_dirs = _scan_run_dirs()
if not run_dirs:
    st.info("还没有历史分析结果。请先回到主页上传 CSV 并完成一次分析。")
else:
    options = {f"{path.parent.name} / {path.name}": path for path in run_dirs}
    selected_label = st.selectbox("选择分析批次", list(options), index=0)
    selected_dir = options[selected_label]
    st.caption(f"数据来源：`{selected_dir}`")

    if st.button("生成该次分析的深度汇总报告", type="primary"):
        try:
            with st.spinner("正在整合报告与图表标注，并调用 DeepSeek 生成总结……"):
                report_text, charts, dataset = _load_run_material(selected_dir)
                summary = generate_executive_summary(report_text, charts, dataset)
            st.session_state.setdefault("exec_summaries", {})[str(selected_dir)] = summary
        except Exception as exc:
            logger.exception("executive summary failed")
            st.error(f"生成失败：{exc}")

    summaries = st.session_state.get("exec_summaries", {})
    summary = summaries.get(str(selected_dir))
    if summary:
        st.success(f"已生成：{selected_label}")
        st.markdown(summary)
        st.download_button(
            "⬇️ 下载 Markdown 报告",
            data=summary,
            file_name=f"executive_summary_{selected_dir.parent.name}_{selected_dir.name}.md",
            mime="text/markdown",
        )
    else:
        st.caption("生成后可在此查看并下载 Markdown 报告。")

# ── 2. 语义搜索历史图表 ────────────────────────────────────────────────────

st.divider()
st.subheader("🔍 用一句话搜索历史图表")
st.caption("基于 PostgreSQL + pgvector + 本地 BGE（sentence-transformers）的语义召回。")

if not settings.DATABASE_URL.strip():
    st.warning("尚未配置 DATABASE_URL，语义搜索不可用。")
    st.code(
        "DATABASE_URL=postgresql://autoviz:autoviz@127.0.0.1:5433/autoviz\n"
        "EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5\n"
        "EMBEDDING_DIM=512",
        language="bash",
    )
    st.info("启动数据库后执行 `uv run autoviz db init`，再运行分析或 `uv run autoviz db ingest <run_dir>`。")
else:
    with st.sidebar:
        st.header("检索配置")
        st.caption(f"向量模型：`{settings.EMBEDDING_MODEL}`（{settings.EMBEDDING_DIM} 维）")
        top_k = st.slider("最终展示条数", min_value=1, max_value=10, value=settings.RAG_TOP_K)
        st.caption(f"召回 {settings.RAG_RECALL_K} 条 → CrossEncoder 重排 → Top {top_k}")
        if settings.RERANK_ENABLED:
            st.caption(f"重排模型：`{settings.RERANK_MODEL}`")
        else:
            st.caption("重排已关闭（RERANK_ENABLED=false）")
        if st.session_state.get("rerank_unavailable"):
            st.warning("重排模型当前不可用，正在使用纯向量检索。")
            if st.button("重试加载重排模型"):
                st.session_state["rerank_unavailable"] = False
                _reranker.clear()
                st.rerun()
        try:
            stats = _store().stats()
            st.success(f"库内：{stats.get('charts', 0)} 张图表 / {stats.get('runs', 0)} 次运行")
        except Exception as exc:
            st.error(f"数据库连接失败：{exc}")

    query = st.text_input(
        "检索问题",
        placeholder="例如：价格和年龄的关系 / 各品类的平均消费 / 性别差异",
    )

    if st.button("搜索历史图表", disabled=not query.strip()):
        removed = _auto_prune(force=True)
        if removed:
            st.caption(f"🧹 搜索前已自动清理 {removed} 条失效历史记录")

        candidates = []
        try:
            with st.spinner("加载向量模型，召回候选图表……"):
                candidates = _store().search(
                    _embedder().encode([query])[0],
                    limit=settings.RAG_RECALL_K,
                )
        except Exception as exc:
            st.error(f"检索失败：{exc}")

        results: list[tuple] = []
        mode = "vector"

        if (
            candidates
            and settings.RERANK_ENABLED
            and not st.session_state.get("rerank_unavailable")
        ):
            try:
                with st.spinner(
                    f"正在对 {len(candidates)} 个候选做 CrossEncoder 重排……"
                ):
                    reranked = rerank_hits(
                        query,
                        candidates,
                        _reranker(),
                        top_k=int(top_k),
                    )
                results = [
                    (hit, score, relevance_probability(score))
                    for hit, score in reranked
                ]
                mode = "rerank"
            except Exception as exc:
                st.session_state["rerank_unavailable"] = True
                logger.warning("reranker failed: %s", exc)
                st.warning(f"重排模型不可用，已自动降级为向量检索：{exc}")

        if candidates and not results:
            results = [(hit, hit.score, None) for hit in candidates[: int(top_k)]]

        if not candidates:
            st.info("没有匹配结果。请先写入向量库（`autoviz db ingest <run_dir>`）。")
        else:
            label = "CrossEncoder 重排" if mode == "rerank" else "向量检索"
            st.subheader(f"Top {len(results)} 结果 · {label}")
            for hit, raw_score, prob in results:
                with st.container(border=True):
                    image_col, text_col = st.columns([1.1, 2.0])
                    with image_col:
                        image_path, debug_path = _resolve_image_path(hit)
                        if image_path:
                            st.image(str(image_path), width="stretch")
                        else:
                            st.warning("历史图片已被清理")
                            with st.expander("路径详情"):
                                st.caption(f"解析路径：`{debug_path}`")
                    with text_col:
                        st.markdown(f"### {hit.title or hit.chart_type}")
                        st.caption(
                            f"{hit.chart_type} · {', '.join(hit.features)} · 数据集 {hit.dataset_name}"
                        )
                        if prob is None:
                            st.progress(
                                max(0.0, min(1.0, float(raw_score))),
                                text=f"余弦相似度 {float(raw_score):.3f}",
                            )
                        else:
                            st.progress(
                                max(0.0, min(1.0, prob)),
                                text=f"重排相关性 {prob:.3f}",
                            )
                            st.caption(f"向量相似度 {float(raw_score):.3f}")
                        if hit.annotation:
                            with st.expander("查看标注"):
                                st.write(hit.annotation)
                        if hit.output_dir:
                            st.caption(f"输出目录：`{hit.output_dir}`")
