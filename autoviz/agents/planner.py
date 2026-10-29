"""Visualization main (planning) agent.

Reads the CSV summary and produces a structured list of VizTaskSpec objects,
each describing one chart to generate. The main agent runs once per pipeline
invocation and drives the sub-agent spawning.
"""

from pydantic import BaseModel
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
from autoviz.agents.prompts import VIZ_MAIN_AGENT_SYSTEM_PROMPT
from autoviz.config import settings
from autoviz.pipeline.csv import CsvSummary


class VizTaskSpec(BaseModel):
    """Single visualization task specification."""

    chart_type: str
    features: list[str]
    title: str
    description: str


class VizPlan(BaseModel):
    """Full visualization plan returned by the main agent."""

    tasks: list[VizTaskSpec]


def _build_main_agent(target_count: int | None = None) -> Agent[None, VizPlan]:
    """Instantiate the main planning agent (uses per-request overrides when set)."""
    model = OpenAIChatModel(
        get_effective_model(),
        provider=OpenAIProvider(
            api_key=get_effective_api_key(),
            base_url="https://api.deepseek.com/v1",  # 强制指向 DeepSeek
        ),
    )
    tc = target_count if target_count is not None else settings.VIZ_TARGET_CHARTS
    system_prompt = VIZ_MAIN_AGENT_SYSTEM_PROMPT.format(target_count=tc) + get_language_instruction()
    return Agent[None, VizPlan](
        model=model,
        output_type=VizPlan,
        model_settings=ModelSettings(temperature=0.4),
        system_prompt=system_prompt,
    )


# Shared agent instance — created once, reused for every pipeline run.
_main_agent: Agent[None, VizPlan] | None = None


def get_main_agent() -> Agent[None, VizPlan]:
    """Return main agent for default target chart count. Uses cached instance when possible."""
    from autoviz.agents.context import has_llm_override

    if has_llm_override() or get_language_context() != "en":
        return _build_main_agent()
    global _main_agent
    if _main_agent is None:
        _main_agent = _build_main_agent()
    return _main_agent


def _resolve_planning_agent(target_count: int) -> Agent[None, VizPlan]:
    """Use cached agent when target_count matches default; else build with matching system prompt."""
    from autoviz.agents.context import has_llm_override

    if has_llm_override() or get_language_context() != "en" or target_count != settings.VIZ_TARGET_CHARTS:
        return _build_main_agent(target_count)
    return get_main_agent()


def _format_exclude_section(exclude_already: list[tuple[str, list[str]]]) -> str:
    """Format already-generated (chart_type, features) for the planning prompt."""
    if not exclude_already:
        return ""
    lines = [f"- {chart_type} | {', '.join(features)}" for chart_type, features in exclude_already]
    return (
        "\n\n**IMPORTANT: The following visualizations have ALREADY been generated in a previous turn. "
        "Do NOT plan any of these again. Plan entirely different (chart_type, features) combinations:**\n"
        + "\n".join(lines)
        + "\n"
    )


def _format_prior_critic_failures(prior_critic_failures: list[dict]) -> str:
    """Format critic rejections so the planner avoids repeating failed (chart_type, features)."""
    if not prior_critic_failures:
        return ""
    lines = []
    for item in prior_critic_failures:
        chart_type = item.get("chart_type", "")
        features = item.get("features") or []
        feats_str = ", ".join(features) if isinstance(features, list) else str(features)
        failure_type = item.get("failure_type", "")
        action = item.get("action", "")
        reason = item.get("critic_reason", "")
        lines.append(
            f"- chart_type={chart_type} | features=[{feats_str}] | "
            f"failure_type={failure_type} | action={action} | reason: {reason}"
        )
    return (
        "\n\n**CRITIC REJECTIONS (this session): These combinations failed quality review. "
        "Do NOT plan the same (chart_type, features) again. Choose different columns and/or chart types "
        "and avoid repeating the same analytical mistake:**\n" + "\n".join(lines) + "\n"
    )


def _build_planning_prompt(
    csv_summary: CsvSummary,
    exclude_already: list[tuple[str, list[str]]] | None = None,
    target_charts: int | None = None,
    prior_critic_failures: list[dict] | None = None,
    user_preferences: str | None = None,
) -> str:
    """Construct the user prompt for the planning agent."""
    count = target_charts if target_charts is not None else settings.VIZ_TARGET_CHARTS
    base = (
        f"Plan {count} visualizations for the following dataset.\n\n" + csv_summary.to_prompt_text()
    )
    if user_preferences:
        base = f"{user_preferences}\n\n{base}"
    if exclude_already:
        base += _format_exclude_section(exclude_already)
    if prior_critic_failures:
        base += _format_prior_critic_failures(prior_critic_failures)
    return base


async def plan_visualizations(
    csv_summary: CsvSummary,
    exclude_already: list[tuple[str, list[str]]] | None = None,
    target_charts: int | None = None,
    prior_critic_failures: list[dict] | None = None,
    user_preferences: str | None = None,
) -> list[VizTaskSpec]:
    """Use the main agent to plan a diverse set of visualizations.

    Args:
        csv_summary: Structured summary of the enhanced CSV.
        exclude_already: Optional list of (chart_type, features) already generated
                         in previous turns; the planner will avoid these.
        target_charts: Optional override for number of charts to plan.
        prior_critic_failures: Rejected tasks with critic_reason / failure_type for this session.
        user_preferences: Optional formatted Q&A context from the MCQ phase.

    Returns:
        List of VizTaskSpec (up to target_charts or VIZ_TARGET_CHARTS items).
    """
    exclude = exclude_already or []
    count = target_charts if target_charts is not None else settings.VIZ_TARGET_CHARTS
    agent = _resolve_planning_agent(count)
    prompt = _build_planning_prompt(
        csv_summary, exclude, target_charts, prior_critic_failures=prior_critic_failures,
        user_preferences=user_preferences,
    )
    result = await agent.run(prompt)
    tasks = result.output.tasks
    return tasks
