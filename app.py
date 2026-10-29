"""AutoViz — Streamlit 本地 Web 界面。

把 CLI（``autoviz analyze run``）包成图形化外壳：
上传 CSV → 侧边栏配置 → subprocess 调用 CLI → 读取最新输出目录 → 渲染报告与图表。
CLI 通过 Popen 流式输出，页面实时展示进度条、完成数量和预计剩余时间。

启动方式::

    uv run streamlit run app.py
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "autoviz-output"
UPLOAD_DIR = PROJECT_ROOT / "workspace" / "uploads"

RUN_TIMEOUT_SECONDS = 3600
MAX_CHARTS = 20
DEFAULT_CHARTS = 3


# ── CLI 调用 ──────────────────────────────────────────────────────────────────


def build_cli_command(csv_path: Path, charts: int, feature_engineering: bool) -> list[str]:
    """构造 CLI 命令（与 README 中的用法一致）。

    默认通过 ``uv run autoviz`` 调用；若环境里没有 uv / autoviz 入口，
    则依次回退到虚拟环境入口和 ``python -m autoviz.cli``。
    """
    args = ["analyze", "run", str(csv_path), "-n", str(charts), "--report"]
    if not feature_engineering:
        args.append("--no-feature-eng")

    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "autoviz", *args]

    entry_point = shutil.which("autoviz")
    if entry_point:
        return [entry_point, *args]

    return [sys.executable, "-m", "autoviz.cli", *args]


def _subprocess_env() -> dict[str, str]:
    """子进程环境：强制 UTF-8，避免 Windows 控制台编码把中文日志变成乱码。"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_GENERATING_RE = re.compile(r"up to\s+(\d+)\s+chart", re.IGNORECASE)
_FINISHED_TASK_RE = re.compile(r"^\s*[+-]\s+.*->")
_DONE_SUMMARY_RE = re.compile(r"Done:\s*(\d+)\s+chart", re.IGNORECASE)


class ProgressTracker:
    """Parse AutoViz CLI output into progress + ETA for the Streamlit UI."""

    def __init__(self, target_charts: int, started_at: float | None = None) -> None:
        self.target = max(1, int(target_charts))
        self.completed = 0
        self.phase = "starting"
        self.progress = 0.02
        self.started_at = started_at if started_at is not None else time.monotonic()
        self.gen_started: float | None = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def text(self) -> str:
        if self.phase == "done":
            return "分析完成"
        if self.phase == "failed":
            return "运行失败"
        if self.phase == "generating":
            return f"正在生成图表 {min(self.completed, self.target)}/{self.target}"
        if self.phase == "report":
            return "正在生成分析报告…"
        if self.phase == "vector":
            return "正在写入向量库…"
        if self.phase == "feature":
            return "正在做 AI 特征工程…"
        if self.phase == "parsing":
            return "正在解析 CSV…"
        return "正在启动分析…"

    def _eta_seconds(self) -> float | None:
        if self.gen_started is None or self.completed <= 0:
            return None
        if self.completed >= self.target:
            return 0.0
        per_chart = (time.monotonic() - self.gen_started) / self.completed
        return max(1.0, per_chart * (self.target - self.completed))

    @property
    def detail(self) -> str:
        elapsed = int(self.elapsed)
        if self.phase == "done":
            return f"总耗时 {elapsed} 秒"
        if self.phase == "failed":
            return f"运行失败 · 已耗时 {elapsed} 秒"
        if self.phase == "generating":
            eta = self._eta_seconds()
            if eta is None:
                return (
                    f"正在生成第 {min(self.completed + 1, self.target)}/{self.target} 张"
                    f" · 已耗时 {elapsed} 秒"
                )
            if eta <= 0:
                return f"图表已全部完成 · 已耗时 {elapsed} 秒"
            return (
                f"已完成 {self.completed}/{self.target} 张"
                f" · 预计剩余约 {int(eta)} 秒 · 已耗时 {elapsed} 秒"
            )
        return f"正在生成中，已耗时 {elapsed} 秒…"

    def handle_line(self, line: str) -> None:
        clean = _ANSI_RE.sub("", line).strip()
        if not clean:
            return

        if "Output directory" in clean:
            self.progress = max(self.progress, 0.03)
        elif "Parsing CSV" in clean:
            self.phase = "parsing"
            self.progress = max(self.progress, 0.04)
        elif "CSV parsed" in clean:
            self.progress = max(self.progress, 0.06)
        elif "Feature engineering" in clean:
            self.phase = "feature"
            self.progress = max(self.progress, 0.08)
        elif "Generating up to" in clean:
            match = _GENERATING_RE.search(clean)
            if match:
                self.target = max(1, int(match.group(1)))
            self.phase = "generating"
            self.gen_started = time.monotonic()
            self.progress = max(self.progress, 0.10)
        elif self.phase == "generating" and _FINISHED_TASK_RE.match(clean):
            self.completed += 1
            ratio = min(self.completed, self.target) / self.target
            self.progress = max(self.progress, 0.10 + 0.78 * ratio)
        elif "Done:" in clean:
            match = _DONE_SUMMARY_RE.search(clean)
            if match:
                self.completed = int(match.group(1))
            self.phase = "report"
            self.progress = max(self.progress, 0.90)
        elif "Generating report" in clean:
            self.phase = "report"
            self.progress = max(self.progress, 0.92)
        elif "Report saved" in clean:
            self.progress = max(self.progress, 0.95)
        elif "Chinese report saved" in clean:
            self.progress = max(self.progress, 0.98)
        elif "向量库已写入" in clean or "vector store" in clean.lower():
            self.phase = "vector"
            self.progress = max(self.progress, 0.99)

    def finish(self) -> None:
        self.phase = "done"
        self.progress = 1.0

    def fail(self) -> None:
        self.phase = "failed"


def stream_cli(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    tracker: ProgressTracker,
    on_line,
    on_update,
) -> tuple[int, bool, list[str]]:
    """Run the CLI with Popen and stream its output line-by-line.

    A background reader thread feeds a queue so the Streamlit script can tick the
    elapsed-time UI even when the child process is silent (e.g. waiting on an LLM).

    Returns:
        (returncode, timed_out, output_lines)
    """
    lines: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=_subprocess_env(),
    )

    output_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                output_queue.put(raw_line.rstrip("\n"))
        finally:
            output_queue.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    timed_out = False

    def _kill() -> None:
        nonlocal timed_out
        timed_out = True
        try:
            process.kill()
        except OSError:
            pass

    timer = threading.Timer(timeout, _kill)
    timer.daemon = True
    timer.start()

    try:
        while True:
            try:
                item = output_queue.get(timeout=0.6)
            except queue.Empty:
                on_update()
                continue

            if item is None:
                break

            lines.append(item)
            tracker.handle_line(item)
            on_line(item)
            on_update()

        return process.wait(), timed_out, lines
    finally:
        timer.cancel()
        if process.poll() is None:
            _kill()
            process.wait()


# ── 文件与输出目录 ────────────────────────────────────────────────────────────


def save_upload(uploaded_file) -> Path:
    """把上传的 CSV 落盘到 ``workspace/uploads/``，返回路径。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w.\-]+", "_", Path(uploaded_file.name).name) or "upload.csv"
    if not safe_name.lower().endswith(".csv"):
        safe_name += ".csv"
    target = UPLOAD_DIR / safe_name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def existing_run_dirs() -> set[Path]:
    """列出 ``autoviz-output/<csv名>/<时间戳>/`` 形式的运行目录。"""
    if not OUTPUT_ROOT.exists():
        return set()
    return {path for path in OUTPUT_ROOT.glob("*/*") if path.is_dir()}


def latest_run_dir(before: set[Path]) -> Path | None:
    """找出本次运行新产生的最新目录；没有新目录则返回 None。"""
    new_dirs = existing_run_dirs() - before
    if not new_dirs:
        return None
    return max(new_dirs, key=lambda path: path.stat().st_mtime)


def _environment_status() -> list[tuple[str, bool]]:
    """检查 .env 中是否配置了关键密钥（只显示是否配置，不显示值）。"""
    try:
        from autoviz.config import settings
    except Exception as exc:  # pragma: no cover - 配置缺失时的兜底
        return [(f"读取配置失败：{exc}", False)]

    return [
        ("DEEPSEEK_API_KEY", bool(settings.DEEPSEEK_API_KEY)),
        ("MODAL_TOKEN_ID", bool(settings.MODAL_TOKEN_ID)),
        ("MODAL_TOKEN_SECRET", bool(settings.MODAL_TOKEN_SECRET)),
    ]


# ── 结果渲染 ──────────────────────────────────────────────────────────────────


def render_results(run_dir: Path) -> None:
    """渲染一次运行的结果：优先中文报告，其次英文报告 + charts/*.png。"""
    st.header("生成结果")
    st.caption(f"输出目录：`{run_dir}`")

    chinese_report = run_dir / "report_chn.md"
    english_report = run_dir / "report.md"
    if chinese_report.exists() and chinese_report.stat().st_size > 0:
        report_path = chinese_report
    else:
        report_path = english_report

    if report_path.exists():
        st.caption(f"报告文件：`{report_path.name}`")
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.warning("未找到 report_chn.md 或 report.md（报告生成可能失败，详见运行日志）。")

    st.divider()

    charts_dir = run_dir / "charts"
    pngs = sorted(charts_dir.glob("*.png")) if charts_dir.is_dir() else []
    if not pngs:
        st.warning("charts/ 目录下没有 PNG 文件。")
        return

    st.subheader(f"图表 · {len(pngs)} 张")
    columns = st.columns(2)
    for index, png in enumerate(pngs):
        with columns[index % 2]:
            st.image(str(png), caption=png.stem, width="stretch")


def _render_previous_run() -> None:
    """从 session_state 恢复上一次运行的结果（窗口重跑后仍然可见）。"""
    run_dir = st.session_state.get("last_run_dir")
    if run_dir:
        render_results(Path(run_dir))


# ── 页面 ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="AutoViz", page_icon="📊", layout="wide")

    st.title("📊 AutoViz")
    st.caption("上传 CSV → 自动规划并绘制图表 → 生成分析报告")

    with st.sidebar:
        st.header("运行配置")
        charts = st.slider(
            "生成图表数量",
            min_value=1,
            max_value=MAX_CHARTS,
            value=DEFAULT_CHARTS,
            step=1,
            help="对应 CLI 的 -n 参数。",
        )
        feature_engineering = st.checkbox(
            "启用 AI 特征工程",
            value=False,
            help=(
                "勾选：额外调用 LLM 生成衍生特征（更慢，需要多一次沙箱执行）；"
                "不勾选：等价于 CLI 的 --no-feature-eng。"
            ),
        )

        st.divider()
        st.subheader("环境检查")
        for name, configured in _environment_status():
            st.write(("✅ " if configured else "⚠️ ") + name + ("" if configured else " 未配置"))
        st.caption("配置读取自项目根目录 `.env`。")

        st.divider()
        st.subheader("将执行")
        st.code(
            " ".join(
                build_cli_command(
                    Path("<上传的文件>.csv"), int(charts), bool(feature_engineering)
                )
            ),
            language="bash",
        )

    uploaded = st.file_uploader("选择 CSV 文件", type=["csv"], accept_multiple_files=False)
    start = st.button("开始 AI 分析", type="primary", disabled=uploaded is None)

    if uploaded is None:
        st.info("请先在左侧设置参数，然后上传 CSV 文件。")

    if start and uploaded is not None:
        csv_path = save_upload(uploaded)
        command = build_cli_command(csv_path, int(charts), bool(feature_engineering))
        before = existing_run_dirs()

        progress_slot = st.empty()
        detail_slot = st.empty()
        tracker = ProgressTracker(charts)

        def refresh_progress() -> None:
            progress_slot.progress(
                max(0.0, min(1.0, tracker.progress)),
                text=tracker.text,
            )
            detail_slot.caption(tracker.detail)

        refresh_progress()

        try:
            returncode, timed_out, log_lines = stream_cli(
                command,
                cwd=PROJECT_ROOT,
                timeout=RUN_TIMEOUT_SECONDS,
                tracker=tracker,
                on_line=lambda line: None,
                on_update=refresh_progress,
            )
        except Exception as exc:  # show launch failures in the page
            returncode, timed_out, log_lines = -1, False, [f"启动 CLI 失败：{exc}"]

        if timed_out:
            tracker.fail()
            st.error(f"运行超过 {RUN_TIMEOUT_SECONDS} 秒，已强制终止。")
        elif returncode != 0:
            tracker.fail()
            st.error(f"CLI 执行失败，退出码 {returncode}。请查看下方日志。")
        else:
            tracker.finish()
            st.success("分析完成。")
        refresh_progress()

        with st.expander("运行日志", expanded=returncode != 0):
            st.code(f"$ {' '.join(command)}\n\n{chr(10).join(log_lines) or '(无输出)'}", language="text")

        run_dir = latest_run_dir(before)
        st.session_state["last_run_dir"] = str(run_dir) if run_dir else None
        if run_dir is None:
            st.warning("本次运行没有产生新的输出目录，未找到可展示的结果。")

    _render_previous_run()


if __name__ == "__main__":
    main()
