"""Visualization agent manager (file-based mode)."""

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from autoviz.agents.annotator import generate_chart_annotation
from autoviz.agents.coder import generate_viz_code
from autoviz.agents.critic import critique_chart
from autoviz.config import settings
from autoviz.events import MessageBus, VizResult, VizTask
from autoviz.pipeline.csv import CsvSummary
from autoviz.pipeline.sandbox import run_visualization_code

logger = logging.getLogger(__name__)

_SEPARATOR = " | "
_STATUS_IN_PROGRESS = "in_progress"
_STATUS_DONE = "done"
_STATUS_ERROR = "error"
_STATUS_SKIPPED = "skipped"


def _features_key(features: list[str]) -> str:
    return "+".join(sorted(features))


def _make_history_row(
    task_id: str,
    chart_type: str,
    features: list[str],
    status: str,
    path: str,
) -> str:
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
    fkey = _features_key(features)
    return f"## [{ts}]{_SEPARATOR}{task_id}{_SEPARATOR}{chart_type}{_SEPARATOR}{fkey}{_SEPARATOR}{status}{_SEPARATOR}{path}\n"


def _parse_history_rows(content: str) -> list[dict]:
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("## ["):
            continue
        try:
            body = line[3:]
            parts = body.split(_SEPARATOR)
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "timestamp": parts[0].strip("[]"),
                    "task_id": parts[1],
                    "chart_type": parts[2],
                    "features_key": parts[3],
                    "status": parts[4],
                    "path": parts[5],
                }
            )
        except Exception:
            continue
    return rows


class HistoryManager:
    """File-based deduplication state."""

    def __init__(self, history_file: Path) -> None:
        self._path = history_file
        self._lock = asyncio.Lock()
        history_file.parent.mkdir(parents=True, exist_ok=True)
        if not history_file.exists():
            history_file.write_text(
                "# Visualization History\n\n"
                "| Timestamp | TaskID | ChartType | Features | Status | Path |\n"
                "|-----------|--------|-----------|----------|--------|------|\n\n",
                encoding="utf-8",
            )

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def _read(self) -> str:
        return self._path.read_text(encoding="utf-8")

    def _append(self, row: str) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(row)

    def _replace_status(self, task_id: str, new_status: str, new_path: str) -> None:
        content = self._read()
        lines = content.splitlines(keepends=True)
        updated = []
        for line in lines:
            if f"{_SEPARATOR}{task_id}{_SEPARATOR}" in line and _STATUS_IN_PROGRESS in line:
                parts = line.rstrip("\n").split(_SEPARATOR)
                if len(parts) >= 6:
                    parts[4] = new_status
                    parts[5] = new_path
                    line = _SEPARATOR.join(parts) + "\n"
            updated.append(line)
        self._path.write_text("".join(updated), encoding="utf-8")

    async def append_critic_failure_note(
        self,
        chart_type: str,
        features: list[str],
        failure_type: str,
        critic_reason: str,
        action: str,
    ) -> None:
        ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        feats = ", ".join(features)
        block = (
            f"\n### Critic note [{ts}]\n"
            f"- chart_type: {chart_type}\n"
            f"- features: {feats}\n"
            f"- failure_type: {failure_type}\n"
            f"- action: {action}\n"
            f"- critic_reason: {critic_reason}\n\n"
        )
        self._append(block)

    async def is_duplicate(self, chart_type: str, features: list[str]) -> str | None:
        fkey = _features_key(features)
        rows = _parse_history_rows(self._read())
        for row in rows:
            if (
                row["chart_type"] == chart_type
                and row["features_key"] == fkey
                and row["status"] in (_STATUS_DONE, _STATUS_IN_PROGRESS)
            ):
                return row["path"] if row["path"] != "-" else ""
        return None

    async def write_in_progress(self, task_id: str, chart_type: str, features: list[str]) -> None:
        row = _make_history_row(task_id, chart_type, features, _STATUS_IN_PROGRESS, "-")
        self._append(row)

    async def write_done(
        self, task_id: str, chart_type: str, features: list[str], image_path: str
    ) -> None:
        async with self._lock:
            self._replace_status(task_id, _STATUS_DONE, image_path)

    async def write_error(
        self,
        task_id: str,
        error_message: str | None = None,
        metadata_patch: dict | None = None,
    ) -> None:
        async with self._lock:
            self._replace_status(task_id, _STATUS_ERROR, "-")

    async def write_skipped(
        self,
        task_id: str,
        metadata_patch: dict | None = None,
    ) -> None:
        async with self._lock:
            self._replace_status(task_id, _STATUS_SKIPPED, "-")


class SubAgentManager:
    """Manages concurrent visualization sub-agent coroutines (file-based output)."""

    def __init__(
        self,
        bus: MessageBus,
        history_manager: HistoryManager,
        enhanced_csv: str,
        csv_summary: CsvSummary,
        viz_workspace: Path,
        concurrency: int | None = None,
        extra_metadata: dict | None = None,
    ) -> None:
        self._bus = bus
        self._history = history_manager
        self._enhanced_csv = enhanced_csv
        self._csv_summary = csv_summary
        self._workspace = viz_workspace
        self._extra_metadata: dict = extra_metadata or {}
        self._semaphore = asyncio.Semaphore(concurrency or settings.VIZ_CONCURRENCY)
        viz_workspace.mkdir(parents=True, exist_ok=True)

    async def _run_one(
        self,
        task: VizTask,
        on_progress: Callable[[VizResult], None] | None,
    ) -> VizResult:
        async with self._semaphore:
            async with self._history.lock:
                existing = await self._history.is_duplicate(task.chart_type, task.features)
                if existing is not None:
                    result = VizResult(
                        task_id=task.task_id,
                        chart_type=task.chart_type,
                        features=task.features,
                        status="skipped",
                        image_path=existing or None,
                    )
                    await self._bus.publish_result(result)
                    if on_progress:
                        on_progress(result)
                    return result

                await self._history.write_in_progress(task.task_id, task.chart_type, task.features)

            base_meta = {
                "title": task.title,
                "description": task.description,
                "chart_type": task.chart_type,
                "features": task.features,
            }
            max_rounds = settings.VIZ_CRITIC_MAX_CODE_ROUNDS
            code_revision_notes: list[dict] = []
            last_code: str | None = None
            png_bytes: bytes | None = None
            last_critique = None
            critic_unavailable_fallback = False
            outcome: str = "annotate"

            for round_idx in range(max_rounds):
                try:
                    code = await generate_viz_code(
                        chart_type=task.chart_type,
                        features=task.features,
                        title=task.title,
                        description=task.description,
                        csv_summary=self._csv_summary,
                        code_revision_notes=code_revision_notes or None,
                        previous_code=last_code,
                    )
                except Exception as exc:
                    logger.error("SubAgent %s: LLM code generation failed: %s", task.task_id, exc)
                    err_meta = {**base_meta, "critic_reason": str(exc), "failure_type": "other"}
                    await self._history.write_error(
                        task.task_id, error_message=str(exc), metadata_patch=err_meta
                    )
                    result = VizResult(
                        task_id=task.task_id,
                        chart_type=task.chart_type,
                        features=task.features,
                        status="error",
                        error=str(exc),
                        metadata=err_meta,
                    )
                    await self._bus.publish_result(result)
                    if on_progress:
                        on_progress(result)
                    return result

                last_code = code
                try:
                    png_bytes = await run_visualization_code(self._enhanced_csv, code)
                except Exception as exc:
                    logger.error("SubAgent %s: sandbox execution failed: %s", task.task_id, exc)
                    code_revision_notes.append(
                        {
                            "critic_reason": f"Sandbox execution failed: {exc}",
                            "failure_type": "execution_error",
                        }
                    )
                    if round_idx < max_rounds - 1:
                        continue
                    err_meta = {
                        **base_meta,
                        "critic_reason": str(exc),
                        "failure_type": "execution_error",
                        "critic_attempts": list(code_revision_notes),
                    }
                    await self._history.write_error(
                        task.task_id, error_message=str(exc), metadata_patch=err_meta
                    )
                    result = VizResult(
                        task_id=task.task_id,
                        chart_type=task.chart_type,
                        features=task.features,
                        status="error",
                        error=str(exc),
                        metadata=err_meta,
                    )
                    await self._bus.publish_result(result)
                    if on_progress:
                        on_progress(result)
                    return result

                try:
                    last_critique = await critique_chart(
                        chart_type=task.chart_type,
                        features=task.features,
                        title=task.title,
                        description=task.description,
                        csv_summary=self._csv_summary,
                        chart_code=last_code,
                    )
                except Exception as exc:
                    logger.warning(
                        "SubAgent %s: critic failed (accepting chart): %s", task.task_id, exc
                    )
                    critic_unavailable_fallback = True
                    last_critique = None
                    outcome = "annotate"
                    break

                if last_critique.is_valid or last_critique.action == "accept":
                    outcome = "annotate"
                    break

                if last_critique.action in ("change_chart_type", "change_features"):
                    replacement_requested = (
                        task.replan_generation < settings.VIZ_CRITIC_MAX_SEMANTIC_REPLANS
                    )
                    skip_meta = {
                        **base_meta,
                        "skip_kind": "critic_semantic",
                        "critic_reason": last_critique.critic_reason,
                        "failure_type": last_critique.failure_type,
                        "action": last_critique.action,
                        "replacement_requested": replacement_requested,
                        "replan_generation": task.replan_generation,
                    }
                    await self._history.append_critic_failure_note(
                        task.chart_type,
                        task.features,
                        last_critique.failure_type,
                        last_critique.critic_reason,
                        last_critique.action,
                    )
                    await self._history.write_skipped(task.task_id, metadata_patch=skip_meta)
                    result = VizResult(
                        task_id=task.task_id,
                        chart_type=task.chart_type,
                        features=task.features,
                        status="skipped",
                        metadata=skip_meta,
                    )
                    await self._bus.publish_result(result)
                    if on_progress:
                        on_progress(result)
                    return result

                code_revision_notes.append(
                    {
                        "critic_reason": last_critique.critic_reason,
                        "failure_type": last_critique.failure_type,
                    }
                )
                if round_idx < max_rounds - 1:
                    continue

                err_msg = last_critique.critic_reason or "Critic rejected chart after max revisions"
                err_meta = {
                    **base_meta,
                    "critic_reason": last_critique.critic_reason,
                    "failure_type": last_critique.failure_type,
                    "action": last_critique.action,
                    "critic_attempts": list(code_revision_notes),
                }
                await self._history.write_error(
                    task.task_id, error_message=err_msg, metadata_patch=err_meta
                )
                result = VizResult(
                    task_id=task.task_id,
                    chart_type=task.chart_type,
                    features=task.features,
                    status="error",
                    error=err_msg,
                    metadata=err_meta,
                )
                await self._bus.publish_result(result)
                if on_progress:
                    on_progress(result)
                return result

            if outcome != "annotate" or png_bytes is None:
                err_meta = {**base_meta, "critic_reason": "Internal visualization state error"}
                await self._history.write_error(
                    task.task_id,
                    error_message="Visualization pipeline internal error",
                    metadata_patch=err_meta,
                )
                result = VizResult(
                    task_id=task.task_id,
                    chart_type=task.chart_type,
                    features=task.features,
                    status="error",
                    error="Internal visualization state error",
                    metadata=err_meta,
                )
                await self._bus.publish_result(result)
                if on_progress:
                    on_progress(result)
                return result

            metadata = dict(base_meta)
            if code_revision_notes:
                metadata["critic_attempts"] = list(code_revision_notes)
            if critic_unavailable_fallback:
                metadata["critic_reason"] = "Critic unavailable; chart accepted by fallback."
            elif last_critique is not None:
                metadata["critic_reason"] = last_critique.critic_reason

            annotation: str | None = None
            try:
                annotation = await generate_chart_annotation(
                    chart_type=task.chart_type,
                    features=task.features,
                    title=task.title,
                    csv_summary=self._csv_summary,
                    description=task.description,
                    chart_code=last_code,
                )
            except Exception as exc:
                logger.warning(
                    "SubAgent %s: annotation generation failed (continuing): %s", task.task_id, exc
                )

            image_path = self._workspace / f"{task.task_id}.png"
            image_path.write_bytes(png_bytes)
            str_path = str(image_path)
            metadata["path"] = str_path
            if annotation:
                metadata["annotation"] = annotation
            await self._history.write_done(
                task.task_id, task.chart_type, task.features, str_path
            )

            if annotation:
                metadata["annotation"] = annotation
            result = VizResult(
                task_id=task.task_id,
                chart_type=task.chart_type,
                features=task.features,
                status="done",
                image_path=str_path,
                metadata=metadata,
            )
            await self._bus.publish_result(result)
            if on_progress:
                on_progress(result)
            return result

    def spawn(
        self,
        task: VizTask,
        on_progress: Callable[[VizResult], None] | None = None,
    ):
        return self._run_one(task, on_progress)

    async def run_all(
        self,
        tasks: list[VizTask],
        on_progress: Callable[[VizResult], None] | None = None,
    ) -> list[VizResult]:
        coroutines = [self.spawn(task, on_progress) for task in tasks]
        return await asyncio.gather(*coroutines, return_exceptions=False)


def make_task_id() -> str:
    return uuid.uuid4().hex[:12]
