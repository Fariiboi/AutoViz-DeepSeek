"""Visualization sub-agent.

Each sub-agent coroutine calls this module to generate matplotlib/seaborn code
for a specific (chart_type, features) combination. All coroutines share the
same underlying Agent instance to avoid redundant model initialisation.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from autoviz.agents.context import get_effective_api_key, get_effective_model
from autoviz.agents.prompts import VIZ_SUB_AGENT_SYSTEM_PROMPT
from autoviz.pipeline.csv import CsvSummary


def _extract_code(text: str) -> str:
    """Strip markdown fences from LLM response if present."""
    text = text.strip()
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    return text


def _build_sub_agent() -> Agent[None, str]:
    """Instantiate the sub-agent (uses per-request overrides when set)."""
    model = OpenAIChatModel(
        get_effective_model(),
        provider=OpenAIProvider(
            api_key=get_effective_api_key(),
            base_url="https://api.deepseek.com/v1",  # 强制指向 DeepSeek
        ),
    )
    return Agent[None, str](
        model=model,
        model_settings=ModelSettings(temperature=0.2),
        system_prompt=VIZ_SUB_AGENT_SYSTEM_PROMPT,
    )


# One Agent instance shared across all concurrent sub-agent coroutines.
_sub_agent: Agent[None, str] | None = None


def get_sub_agent() -> Agent[None, str]:
    """Return sub-agent. Uses cached instance only when no per-request override is set."""
    from autoviz.agents.context import has_llm_override

    if has_llm_override():
        return _build_sub_agent()
    global _sub_agent
    if _sub_agent is None:
        _sub_agent = _build_sub_agent()
    return _sub_agent


def _build_sub_agent_prompt(
    chart_type: str,
    features: list[str],
    title: str,
    description: str,
    csv_summary: CsvSummary,
    *,
    code_revision_notes: list[dict] | None = None,
    previous_code: str | None = None,
) -> str:
    features_str = ", ".join(features)
    base = (
        f"Chart type: {chart_type}\n"
        f"Features/columns to use: {features_str}\n"
        f"Title: {title}\n"
        f"Description: {description}\n\n"
        f"Dataset summary:\n{csv_summary.to_prompt_text()}"
    )
    if previous_code or code_revision_notes:
        base += "\n\n---\n**Revision context** (fix the chart; previous attempt failed review or execution):\n"
        if code_revision_notes:
            for i, note in enumerate(code_revision_notes, start=1):
                reason = note.get("critic_reason", "")
                ftype = note.get("failure_type", "")
                base += f"\nAttempt {i}: failure_type={ftype}\nCritic / error: {reason}\n"
        if previous_code:
            excerpt = (
                previous_code
                if len(previous_code) <= 12000
                else previous_code[:12000] + "\n# ... truncated ..."
            )
            base += f"\nPrevious Python code (rewrite to fix issues):\n```python\n{excerpt}\n```\n"
    return base


async def generate_viz_code(
    chart_type: str,
    features: list[str],
    title: str,
    description: str,
    csv_summary: CsvSummary,
    *,
    code_revision_notes: list[dict] | None = None,
    previous_code: str | None = None,
) -> str:
    """Generate matplotlib/seaborn visualization code for one chart.

    Args:
        chart_type: e.g. "bar", "scatter", "heatmap"
        features: list of column names to visualise
        title: descriptive chart title
        description: one-sentence insight description
        csv_summary: structured dataset summary for context
        code_revision_notes: prior critic or sandbox errors (dicts with critic_reason, failure_type)
        previous_code: last generated code when revising after failed execute/critic

    Returns:
        Executable Python code string (no markdown fences).
    """
    agent = get_sub_agent()
    prompt = _build_sub_agent_prompt(
        chart_type,
        features,
        title,
        description,
        csv_summary,
        code_revision_notes=code_revision_notes,
        previous_code=previous_code,
    )
    result = await agent.run(prompt)
    code = _extract_code(result.output)
    return code
