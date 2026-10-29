"""Chart-to-text annotation agent (text-only).

DeepSeek chat models accept text only, so the annotator never receives the
rendered PNG. It writes the annotation from the deterministic chart digest built
by :mod:`autoviz.agents.chart_digest`: the planned task, the data profile of the
plotted columns, and the rendering choices extracted from the generated code.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from autoviz.agents.chart_digest import build_chart_digest
from autoviz.agents.context import (
    get_effective_api_key,
    get_effective_model,
    get_language_context,
    get_language_instruction,
)
from autoviz.agents.prompts import VIZ_ANNOTATION_SYSTEM_PROMPT
from autoviz.pipeline.csv import CsvSummary


def _build_annotation_agent() -> Agent[None, str]:
    """Create the annotation agent (text-only model, uses per-request overrides when set)."""
    model = OpenAIChatModel(
        get_effective_model(),
        provider=OpenAIProvider(
            api_key=get_effective_api_key(),
            base_url="https://api.deepseek.com/v1",  # 强制指向 DeepSeek
        ),
    )
    return Agent[None, str](
        model=model,
        model_settings=ModelSettings(temperature=0.3),
        system_prompt=VIZ_ANNOTATION_SYSTEM_PROMPT + get_language_instruction(),
    )


_annotation_agent: Agent[None, str] | None = None


def _get_annotation_agent() -> Agent[None, str]:
    """Return annotation agent. Uses cached instance only when no per-request override is set."""
    from autoviz.agents.context import has_llm_override

    if has_llm_override() or get_language_context() != "en":
        return _build_annotation_agent()
    global _annotation_agent
    if _annotation_agent is None:
        _annotation_agent = _build_annotation_agent()
    return _annotation_agent


def _build_annotation_prompt(
    chart_type: str,
    features: list[str],
    title: str,
    description: str | None,
    csv_summary: CsvSummary,
    chart_code: str | None,
) -> str:
    digest = build_chart_digest(
        chart_type=chart_type,
        features=features,
        title=title,
        description=description,
        csv_summary=csv_summary,
        chart_code=chart_code,
    )
    return (
        digest.to_prompt_text()
        + "\n\n---\nWrite the annotation for this chart from the digest above."
    )


async def generate_chart_annotation(
    chart_type: str,
    features: list[str],
    title: str,
    csv_summary: CsvSummary,
    description: str | None = None,
    chart_code: str | None = None,
) -> str:
    """Generate a rich annotation (chart-to-text) for a visualization, from text only.

    Args:
        chart_type: e.g. bar, scatter, heatmap.
        features: Column names used in the chart.
        title: Chart title.
        csv_summary: Dataset summary for context.
        description: Planned one-sentence insight (optional).
        chart_code: Code that produced the chart; parsed for axes/scales.

    Returns:
        Annotation text (2-4 paragraphs).
    """
    agent = _get_annotation_agent()
    text_prompt = _build_annotation_prompt(
        chart_type, features, title, description, csv_summary, chart_code
    )
    result = await agent.run(text_prompt)
    annotation = result.output.strip()
    return annotation
