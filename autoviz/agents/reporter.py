"""Report generation agent.

After visualizations and annotations are complete, this agent synthesizes
CsvSummary and visualization properties (title, chart_type, features, annotation)
into a structured report: dataset overview, theme-aggregated key findings,
relationship analysis, and suggestions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from autoviz.agents.context import (
    get_effective_api_key,
    get_effective_model,
    get_language_context,
    get_language_instruction,
)
from autoviz.agents.prompts import REPORT_AGENT_SYSTEM_PROMPT
from autoviz.pipeline.csv import CsvSummary


@dataclass
class VizProperty:
    """Properties of a single visualization for report context."""

    title: str
    chart_type: str
    features: list[str]
    annotation: str | None



def _parse_viz_refs(text: str) -> list[int]:
    """Extract 1-indexed viz numbers from text like '(Visualizations 1, 11)'."""
    refs: set[int] = set()
    for m in re.finditer(r"Visualizations?\s+([\d,\s]+)", text, re.IGNORECASE):
        for n in re.findall(r"\d+", m.group(1)):
            refs.add(int(n))
    return sorted(refs)


def _strip_viz_ref(text: str) -> str:
    """Remove parenthetical viz reference from text."""
    return re.sub(r"\s*\(Visualizations?\s+[\d,\s]+\)\s*\.?", "", text, flags=re.IGNORECASE).strip()


class KeyFindingDetail(BaseModel):
    """A single finding with optional visualization references."""

    text: str
    viz_refs: list[int] = []  # 1-indexed visualization numbers (e.g. 1 = first chart)


class KeyFindingTheme(BaseModel):
    """A thematic grouping of key findings (not per-chart)."""

    theme: str
    summary: str
    details: list[KeyFindingDetail] = []

    @field_validator("details", mode="before")
    @classmethod
    def _normalize_details(cls, v: Any) -> list[dict[str, Any]]:
        """Accept legacy format: list of strings with '(Visualization N)' in text."""
        if not v:
            return []
        result: list[dict[str, Any]] = []
        for item in v:
            if isinstance(item, str):
                refs = _parse_viz_refs(item)
                text = _strip_viz_ref(item) if refs else item
                result.append({"text": text, "viz_refs": refs})
            elif isinstance(item, dict):
                result.append(item)
            else:
                result.append({"text": str(item), "viz_refs": []})
        return result


class RelationshipInsight(BaseModel):
    """An insight about relationships among variables."""

    variables: list[str]
    description: str


class DataReport(BaseModel):
    """Structured report output from the report agent."""

    dataset_overview: str
    key_findings: list[KeyFindingTheme]
    relationship_analysis: list[RelationshipInsight]
    suggestions: list[str]


def _build_report_agent() -> Agent[None, DataReport]:
    """Create the report agent (uses per-request overrides when set)."""
    model = OpenAIChatModel(
        get_effective_model(),
        provider=OpenAIProvider(
            api_key=get_effective_api_key(),
            base_url="https://api.deepseek.com/v1",  # 强制指向 DeepSeek
        ),
    )
    return Agent[None, DataReport](
        model=model,
        output_type=DataReport,
        model_settings=ModelSettings(temperature=0.4),
        system_prompt=REPORT_AGENT_SYSTEM_PROMPT + get_language_instruction(),
    )


_report_agent: Agent[None, DataReport] | None = None


def _get_report_agent() -> Agent[None, DataReport]:
    """Return report agent. Uses cached instance only when no per-request override is set."""
    from autoviz.agents.context import has_llm_override

    if has_llm_override() or get_language_context() != "en":
        return _build_report_agent()
    global _report_agent
    if _report_agent is None:
        _report_agent = _build_report_agent()
    return _report_agent


def _build_report_prompt(
    csv_summary: CsvSummary,
    viz_properties: list[VizProperty],
) -> str:
    """Build the user prompt for the report agent."""
    lines = [
        "Generate a structured data analysis report based on the following inputs.",
        "",
        "## Dataset context",
        "",
        csv_summary.to_prompt_text(),
        "",
        "## Visualizations and their annotations",
        "",
    ]
    for i, v in enumerate(viz_properties, 1):
        lines.append(f"### Visualization {i}")
        lines.append(f"- Title: {v.title}")
        lines.append(f"- Chart type: {v.chart_type}")
        lines.append(f"- Features: {', '.join(v.features)}")
        if v.annotation:
            lines.append(f"- Annotation: {v.annotation}")
        else:
            lines.append("- Annotation: (none)")
        lines.append("")

    lines.append(
        "Synthesize the above into a structured report. Aggregate insights by theme, "
        "not by individual chart. Focus on cross-chart patterns and relationships."
    )
    return "\n".join(lines)


async def generate_report(
    csv_summary: CsvSummary,
    viz_properties: list[VizProperty],
) -> DataReport:
    """Generate a structured report from dataset and visualization context.

    Args:
        csv_summary: Summary of the dataset (columns, types, samples).
        viz_properties: List of visualization properties (title, chart_type,
                        features, annotation) for each generated chart.

    Returns:
        DataReport with dataset_overview, theme-aggregated key_findings,
        relationship_analysis, and suggestions.
    """
    agent = _get_report_agent()
    prompt = _build_report_prompt(csv_summary, viz_properties)
    result = await agent.run(prompt)
    report = result.output
    return report
