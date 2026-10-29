"""Chart quality critic agent (text-only, structured output).

DeepSeek chat models accept text only, so the critic never receives the rendered
PNG. It reviews the deterministic chart digest built by
:mod:`autoviz.agents.chart_digest` — the planned task, the data profile of the
plotted columns, and a static reading of the generated code (axis labels,
scales, aggregation, ...).

Consequence: purely visual failure modes (blank canvas, tofu/garbled glyphs)
are no longer detectable and have been removed from :data:`VizFailureType`.
"""

from typing import Literal

from pydantic import BaseModel, Field
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
from autoviz.agents.prompts import VIZ_CRITIC_SYSTEM_PROMPT
from autoviz.pipeline.csv import CsvSummary

# Text-only failure taxonomy. Vision-only values ("empty_plot", "wrong_encoding")
# were dropped together with image input.
VizFailureType = Literal[
    "weak_signal",
    "misleading",
    "execution_error",
    "other",
]

VizCriticAction = Literal["accept", "retry_code", "change_chart_type", "change_features"]


class VizCritique(BaseModel):
    """Structured critic output for routing (code retry vs planner vs accept)."""

    is_valid: bool
    failure_type: VizFailureType = "other"
    action: VizCriticAction
    critic_reason: str = Field(default="", max_length=4000)


def _build_critic_agent() -> Agent[None, VizCritique]:
    model = OpenAIChatModel(
        get_effective_model(),
        provider=OpenAIProvider(
            api_key=get_effective_api_key(),
            base_url="https://api.deepseek.com/v1",  # 强制指向 DeepSeek
        ),
    )
    return Agent[None, VizCritique](
        model=model,
        output_type=VizCritique,
        model_settings=ModelSettings(temperature=0.1),
        system_prompt=VIZ_CRITIC_SYSTEM_PROMPT + get_language_instruction(),
    )


_critic_agent: Agent[None, VizCritique] | None = None


def _get_critic_agent() -> Agent[None, VizCritique]:
    from autoviz.agents.context import has_llm_override

    if has_llm_override() or get_language_context() != "en":
        return _build_critic_agent()
    global _critic_agent
    if _critic_agent is None:
        _critic_agent = _build_critic_agent()
    return _critic_agent


def _build_critic_user_prompt(
    chart_type: str,
    features: list[str],
    title: str,
    description: str,
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
        + "\n\n---\nReview this chart from the digest above and return the structured verdict."
    )


async def critique_chart(
    chart_type: str,
    features: list[str],
    title: str,
    description: str,
    csv_summary: CsvSummary,
    chart_code: str | None = None,
) -> VizCritique:
    """Run the text-only critic on one chart. Returns structured routing fields.

    Args:
        chart_type: Planned chart type.
        features: Columns the chart is supposed to show.
        title: Planned title.
        description: One-sentence intended insight.
        csv_summary: Dataset summary, used to profile the plotted columns.
        chart_code: Code that produced the chart (parsed for axes/scales/etc.).

    Returns:
        VizCritique with routing fields (accept / retry_code / replan).
    """
    agent = _get_critic_agent()
    prompt = _build_critic_user_prompt(
        chart_type, features, title, description, csv_summary, chart_code
    )
    result = await agent.run(prompt)
    return result.output
