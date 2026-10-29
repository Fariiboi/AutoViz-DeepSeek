"""Feature engineering agent.

Generates Python code for derived features based on CSV summary.
Uses CsvSummary as context (not full CSV) to keep token usage low.
"""


from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from autoviz.agents.context import get_effective_api_key, get_effective_model
from autoviz.agents.prompts import FEATURE_ENGINEER_SYSTEM_PROMPT
from autoviz.pipeline.csv import CsvSummary


def _extract_code_from_response(text: str) -> str:
    """Extract executable Python code from LLM response.

    Handles markdown code blocks if the model wraps code in ```python ... ```
    """
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


def build_feature_engineer_prompt(
    csv_summary: CsvSummary, user_preferences: str | None = None
) -> str:
    """Build user prompt from CSV summary and optional user preferences."""
    base = csv_summary.to_prompt_text()
    if user_preferences:
        return f"{user_preferences}\n\n{base}"
    return base


def get_feature_engineer_agent() -> Agent[None, str]:
    """Create and return the feature engineering agent (uses per-request overrides when set)."""
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
        system_prompt=FEATURE_ENGINEER_SYSTEM_PROMPT,
    )


async def generate_feature_engineering_code(
    csv_summary: CsvSummary, user_preferences: str | None = None
) -> str:
    """Generate Python code for feature engineering from CSV summary.

    Args:
        csv_summary: Structured summary of the CSV (column types, samples, etc.)
        user_preferences: Optional formatted Q&A context from the MCQ phase.

    Returns:
        Executable Python code string. Assumes df exists and must end with
        df.to_csv('/home/user/output.csv', index=False).
    """
    agent = get_feature_engineer_agent()
    prompt = build_feature_engineer_prompt(csv_summary, user_preferences)
    result = await agent.run(prompt)
    raw = result.output
    code = _extract_code_from_response(raw)
    return code
