"""Translate the English AutoViz report into a Chinese business report."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from autoviz.agents.context import get_effective_api_key, get_effective_model
from autoviz.agents.prompts import REPORT_CHINESE_SYSTEM_PROMPT

MAX_REPORT_CHARS = 30_000


def build_translation_prompt(report_markdown: str) -> str:
    """Build the translation prompt, truncating pathological inputs safely."""
    report = (report_markdown or "").strip()
    truncated = False
    if len(report) > MAX_REPORT_CHARS:
        report = report[:MAX_REPORT_CHARS]
        truncated = True

    prompt = (
        "Translate the following English Markdown report into a high-quality Simplified "
        "Chinese business analysis report. Preserve every section, number and "
        "'Visualization N' reference, keep the Markdown structure, and do not add or remove "
        "content. Return Markdown only.\n\n"
        f"{report}"
    )
    if truncated:
        prompt += "\n\n(The English report was truncated because it exceeded the input limit.)"
    return prompt


def _build_translation_agent() -> Agent[None, str]:
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
        system_prompt=REPORT_CHINESE_SYSTEM_PROMPT,
    )


_translation_agent: Agent[None, str] | None = None


def _get_translation_agent() -> Agent[None, str]:
    from autoviz.agents.context import has_llm_override

    if has_llm_override():
        return _build_translation_agent()
    global _translation_agent
    if _translation_agent is None:
        _translation_agent = _build_translation_agent()
    return _translation_agent


async def translate_report_to_chinese(report_markdown: str) -> str:
    """Generate the Chinese counterpart of the English report."""
    prompt = build_translation_prompt(report_markdown)
    result = await _get_translation_agent().run(prompt)
    return result.output.strip()
