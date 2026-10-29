"""Executive summary agent for a completed AutoViz analysis batch."""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from autoviz.agents.context import (
    get_effective_api_key,
    get_effective_model,
)
from autoviz.agents.prompts import EXECUTIVE_SUMMARY_SYSTEM_PROMPT

MAX_REPORT_CHARS = 20_000
MAX_ANNOTATION_CHARS = 1_500
MAX_CHARTS = 20


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _format_chart(index: int, chart: dict[str, Any]) -> list[str]:
    title = _text(chart.get("title")) or "未命名图表"
    chart_type = _text(chart.get("chart_type")) or "unknown"
    features = [str(feature) for feature in (chart.get("features") or [])]
    annotation = " ".join(_text(chart.get("annotation")).split())
    if len(annotation) > MAX_ANNOTATION_CHARS:
        annotation = annotation[:MAX_ANNOTATION_CHARS] + "……"

    lines = [f"{index}. 《{title}》（{chart_type}）"]
    if features:
        lines.append(f"   字段：{', '.join(features)}")
    if annotation:
        lines.append(f"   标注：{annotation}")
    return lines


def build_executive_summary_prompt(
    report_markdown: str,
    charts: list[dict[str, Any]],
    dataset: dict[str, Any] | None = None,
) -> str:
    """Build the super-prompt sent to DeepSeek for the executive summary."""
    report = (report_markdown or "").strip()
    truncated = False
    if len(report) > MAX_REPORT_CHARS:
        report = report[:MAX_REPORT_CHARS]
        truncated = True

    lines: list[str] = ["## 一、历史分析报告", "", report or "（本次分析没有生成报告）"]
    if truncated:
        lines.extend(["", "（报告过长，以上内容已截断）"])

    lines.extend(["", f"## 二、图表与标注（共 {len(charts)} 张）", ""])
    if charts:
        for index, chart in enumerate(charts[:MAX_CHARTS], start=1):
            lines.extend(_format_chart(index, chart))
            lines.append("")
        if len(charts) > MAX_CHARTS:
            lines.append(f"（仅展示前 {MAX_CHARTS} 张图表）")
    else:
        lines.append("（本次分析没有可用的图表元数据）")

    if dataset:
        lines.extend(["", "## 三、数据集信息", ""])
        row_count = dataset.get("row_count")
        column_count = dataset.get("column_count")
        sampled_rows = dataset.get("sampled_rows")
        if row_count is not None:
            lines.append(f"- 数据行数：{row_count:,}")
        if column_count is not None:
            lines.append(f"- 字段数量：{column_count}")
        if sampled_rows:
            lines.append(f"- 本次分析使用随机样本：{sampled_rows:,} 行")

    lines.extend(
        [
            "",
            "## 四、任务",
            "",
            "请基于以上详细的图表描述和历史报告，为老板写一份 450-600 字的商业洞察总结。",
            "要求：",
            "1. 总字数控制在 450-600 字，不得少于 450 字；",
            "2. 开头用一句话给出总体判断（约 40 字）；",
            "3. 核心结论：恰好 3 条，每条 70-100 字，按业务影响排序；",
            "4. 行动建议：恰好 2 条，每条 50-80 字；",
            "5. 只使用材料中的事实与数字，不要编造数据；",
            "6. 只输出正文，不要标题，不要使用 # 号标题，避免技术实现细节。",
        ]
    )
    return "\n".join(lines)


def _build_summary_agent() -> Agent[None, str]:
    model = OpenAIChatModel(
        get_effective_model(),
        provider=OpenAIProvider(
            api_key=get_effective_api_key(),
            base_url="https://api.deepseek.com/v1",  # 强制指向 DeepSeek
        ),
    )
    return Agent[None, str](
        model=model,
        model_settings=ModelSettings(temperature=0.4),
        # Always Simplified Chinese: the audience is Chinese business leadership.
        system_prompt=EXECUTIVE_SUMMARY_SYSTEM_PROMPT,
    )


_summary_agent: Agent[None, str] | None = None


def _get_summary_agent() -> Agent[None, str]:
    from autoviz.agents.context import has_llm_override

    if has_llm_override():
        return _build_summary_agent()
    global _summary_agent
    if _summary_agent is None:
        _summary_agent = _build_summary_agent()
    return _summary_agent


def generate_executive_summary(
    report_markdown: str,
    charts: list[dict[str, Any]],
    dataset: dict[str, Any] | None = None,
) -> str:
    """Generate a ~500-character executive summary for one analysis batch."""
    prompt = build_executive_summary_prompt(report_markdown, charts, dataset)
    result = _get_summary_agent().run_sync(prompt)
    return result.output.strip()
