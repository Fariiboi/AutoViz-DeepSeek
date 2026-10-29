"""Feature engineering pipeline.

Orchestrates: LLM code generation from CSV summary -> E2B sandbox execution -> enhanced df.
"""

from collections.abc import Callable

from autoviz.agents.feature_engineer import generate_feature_engineering_code
from autoviz.pipeline.csv import CsvSummary
from autoviz.pipeline.sandbox import run_feature_engineering_code


async def run_feature_engineering(
    csv_summary: CsvSummary,
    preprocessed_csv: str,
    *,
    on_code_generated: Callable[[str], None] | None = None,
    on_stdout: Callable[[str], None] | None = None,
    on_stderr: Callable[[str], None] | None = None,
    user_preferences: str | None = None,
) -> str:
    """Run full feature engineering pipeline.

    1. Generate Python code from CSV summary via LLM.
    2. Execute code in E2B sandbox with preprocessed CSV.
    3. Return enhanced dataframe as CSV string.

    Args:
        csv_summary: Structured summary for LLM context.
        preprocessed_csv: Preprocessed dataframe as CSV string.
        on_code_generated: Optional callback when LLM code is ready.
        on_stdout: Optional callback for sandbox stdout (WebSocket progress).
        on_stderr: Optional callback for sandbox stderr.
        user_preferences: Optional formatted Q&A context from the MCQ phase.

    Returns:
        Enhanced dataframe as CSV string.

    Raises:
        ValueError: If csv_summary has error or preprocessed_csv is empty.
        RuntimeError: If LLM or sandbox execution fails.
    """
    if csv_summary.error:
        raise ValueError(f"CSV summary has error: {csv_summary.error}")
    if not preprocessed_csv.strip():
        raise ValueError("Preprocessed CSV is empty")

    code = await generate_feature_engineering_code(csv_summary, user_preferences)
    if on_code_generated:
        on_code_generated(code)

    output_csv = await run_feature_engineering_code(
        preprocessed_csv,
        code,
        on_stdout=on_stdout,
        on_stderr=on_stderr,
    )
    return output_csv
