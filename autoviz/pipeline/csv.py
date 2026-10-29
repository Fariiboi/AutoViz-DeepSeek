"""CSV preprocessing pipeline for AI agent context.

Preprocesses CSV content to extract summarized information (column types,
sample rows, etc.) before passing to the LLM. Keeps token usage low and
provides structured context.

Pipeline steps:
0. Decode the file (UTF-8 with BOM, falling back to GB18030/GBK) and randomly
   sample large files down to ``CSV_MAX_ANALYSIS_ROWS`` rows
1. Parse CSV
2. Detect and handle missing values (fill_mean for numeric)
3. Detect and remove duplicate rows
4. Extract summary for display and agent context

Extensible: add more steps in preprocess_csv() as needed.
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from autoviz.config import settings

logger = logging.getLogger(__name__)


def read_text_auto(path: Path) -> str:
    """Read a text file, trying UTF-8 (with BOM) first and GB18030 afterwards.

    GB18030 is a superset of GBK/GB2312, which is what Excel on Chinese Windows
    typically produces. Raises ``ValueError`` when neither codec can decode it.
    """
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码（已尝试 utf-8、gb18030）：{path}")


def _is_numeric(series: pd.Series) -> bool:
    """Check if column is numeric (int or float). Uses pandas API to handle extension dtypes."""
    return pd.api.types.is_numeric_dtype(series)


def _to_json_safe(val) -> str | int | float | bool | None:
    """Convert a pandas/numpy value to JSON-serializable type."""
    if pd.isna(val):
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


@dataclass
class CsvSummary:
    """Structured summary of a CSV file for agent context."""

    column_types: dict[str, str] = field(default_factory=dict)
    sample_rows: str = ""
    sample_data: list[dict[str, str | int | float | bool | None]] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    error: str | None = None
    # Preprocessing info
    missing_values_by_column: dict[str, int] = field(default_factory=dict)
    missing_values_handled: str = ""
    duplicate_rows_removed: int = 0
    row_count_before_cleaning: int = 0
    # Sampling info (large-file protection)
    row_count_before_sampling: int = 0
    sampled_rows: int = 0
    # Per-column statistics for Selector layer
    column_unique_counts: dict[str, int] = field(default_factory=dict)
    column_numeric_stats: dict[str, dict] = field(default_factory=dict)  # {col: {std, mean}}

    def to_dict(self) -> dict:
        """Serialize to dict for JSON (e.g. WebSocket)."""
        return {
            "column_types": self.column_types,
            "sample_rows": self.sample_rows,
            "sample_data": self.sample_data,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "error": self.error,
            "missing_values_by_column": self.missing_values_by_column,
            "missing_values_handled": self.missing_values_handled,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "row_count_before_cleaning": self.row_count_before_cleaning,
            "row_count_before_sampling": self.row_count_before_sampling,
            "sampled_rows": self.sampled_rows,
            "column_unique_counts": self.column_unique_counts,
            "column_numeric_stats": self.column_numeric_stats,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CsvSummary":
        """Reconstruct CsvSummary from a dict (e.g. from DB)."""
        return cls(
            column_types=d.get("column_types") or {},
            sample_rows=d.get("sample_rows") or "",
            sample_data=d.get("sample_data") or [],
            row_count=int(d.get("row_count") or 0),
            column_count=int(d.get("column_count") or 0),
            error=d.get("error"),
            missing_values_by_column=d.get("missing_values_by_column") or {},
            missing_values_handled=d.get("missing_values_handled") or "",
            duplicate_rows_removed=int(d.get("duplicate_rows_removed") or 0),
            row_count_before_cleaning=int(d.get("row_count_before_cleaning") or 0),
            row_count_before_sampling=int(d.get("row_count_before_sampling") or 0),
            sampled_rows=int(d.get("sampled_rows") or 0),
            column_unique_counts=d.get("column_unique_counts") or {},
            column_numeric_stats=d.get("column_numeric_stats") or {},
        )

    def to_prompt_text(self) -> str:
        """Format the summary for inclusion in the agent prompt."""
        if self.error:
            return f"CSV could not be parsed: {self.error}"

        lines = [
            "The user has attached a CSV file. Here is a summary:",
            "",
            f"- Rows: {self.row_count:,}",
        ]
        if self.sampled_rows:
            lines.append(
                f"  (random sample of {self.row_count_before_sampling:,} rows; "
                "statistics below describe the sample)"
            )
        lines.extend(
            [
                f"- Columns: {self.column_count:,}",
                "",
                "Column types:",
            ]
        )
        for col, dtype in self.column_types.items():
            lines.append(f"  - {col}: {dtype}")

        if self.missing_values_handled:
            lines.extend(["", "Missing values:", self.missing_values_handled])
        if self.duplicate_rows_removed > 0:
            lines.append(f"Duplicate rows removed: {self.duplicate_rows_removed:,}")

        if self.sample_rows:
            lines.extend(["", "Sample rows (random):", "```", self.sample_rows, "```"])

        return "\n".join(lines)

    def to_display_text(self, title: str = "CSV parsing complete") -> str:
        """Format the summary for frontend display (e.g. markdown)."""
        if self.error:
            return f"❌ CSV parse failed: {self.error}"

        lines = [
            f"📊 **{title}**",
            "",
            f"- **Rows:** {self.row_count:,}",
            f"- **Columns:** {self.column_count:,}",
        ]
        if self.duplicate_rows_removed and self.duplicate_rows_removed > 0:
            lines.append(f"- **Duplicate rows removed:** {self.duplicate_rows_removed:,}")
        if self.missing_values_handled:
            lines.extend(["", "**Missing values handled:**", self.missing_values_handled])
        lines.extend(["", "**Column types:**"])
        for col, dtype in self.column_types.items():
            lines.append(f"  - `{col}`: {dtype}")
        if self.sample_rows:
            lines.extend(["", "**Sample data:**", f"```\n{self.sample_rows}\n```"])
        return "\n".join(lines)


def _dtype_to_readable(dtype) -> str:
    """Convert pandas dtype to a human-readable type name."""
    name = str(dtype)
    mapping = {
        "object": "string",
        "int64": "integer",
        "int32": "integer",
        "float64": "float",
        "float32": "float",
        "bool": "boolean",
        "datetime64[ns]": "datetime",
        "datetime64[us]": "datetime",
    }
    for k, v in mapping.items():
        if k in name:
            return v
    return name


def preprocess_csv_with_data(
    csv_content: str,
    sample_size: int | None = None,
) -> tuple[CsvSummary, str]:
    """Preprocess CSV and return both summary and preprocessed data as CSV string.

    Steps: (optionally) random-sample large inputs, handle missing values, remove
    duplicates. Returns the cleaned dataframe serialized as CSV for downstream
    pipelines (e.g. feature engineering).

    Args:
        csv_content: Already-decoded CSV text (use ``read_text_auto`` for files).
        sample_size: Rows kept in ``CsvSummary.sample_rows``.

    Returns:
        Tuple of (CsvSummary, preprocessed_csv_str). Empty string if parse failed.
    """
    if sample_size is None:
        sample_size = settings.CSV_SAMPLE_SIZE

    try:
        df = pd.read_csv(io.StringIO(csv_content))
    except Exception as e:
        logger.warning("CSV parse failed: %s", e)
        return CsvSummary(error=str(e)), ""

    row_count_before_sampling = len(df)
    sampled_rows = 0
    max_rows = settings.CSV_MAX_ANALYSIS_ROWS
    if max_rows > 0 and row_count_before_sampling > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        sampled_rows = len(df)
        logger.info(
            "Large CSV detected (%s rows): sampled %s rows for analysis",
            f"{row_count_before_sampling:,}",
            f"{sampled_rows:,}",
        )

    row_count_before_cleaning = len(df)
    missing_values_by_column: dict[str, int] = {}
    missing_handled_parts: list[str] = []

    # 1. Detect and handle missing values
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    for col in cols_with_missing.index:
        count = int(missing_counts[col])
        missing_values_by_column[str(col)] = count

        if _is_numeric(df[col]):
            mean_val = df[col].mean()
            if pd.isna(mean_val):
                mean_val = 0
            df[col] = df[col].fillna(mean_val)
            missing_handled_parts.append(f"{col}: {count:,} filled with mean ({mean_val:.4g})")
        else:
            fill_val = "unknown"
            df[col] = df[col].fillna(fill_val)
            missing_handled_parts.append(f"{col}: {count:,} filled with unknown\n")

    missing_values_handled = "; ".join(missing_handled_parts) if missing_handled_parts else ""

    # 2. Detect and remove duplicate rows
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        df = df.drop_duplicates(ignore_index=True)

    row_count = len(df)
    column_count = len(df.columns)

    column_types = {str(col): _dtype_to_readable(df[col].dtype) for col in df.columns}

    # Per-column statistics for the Selector layer
    column_unique_counts: dict[str, int] = {}
    column_numeric_stats: dict[str, dict] = {}
    for col in df.columns:
        col_str = str(col)
        try:
            column_unique_counts[col_str] = int(df[col].nunique(dropna=True))
        except Exception:
            column_unique_counts[col_str] = 0
        if _is_numeric(df[col]) and row_count > 0:
            try:
                std_val = float(df[col].std())
                mean_val = float(df[col].mean())
                column_numeric_stats[col_str] = {
                    "std": std_val if not pd.isna(std_val) else 0.0,
                    "mean": mean_val if not pd.isna(mean_val) else 0.0,
                }
            except Exception:
                pass

    if row_count == 0:
        sample_str = "(no rows)"
        sample_data: list[dict[str, str | int | float | bool | None]] = []
    else:
        n = min(sample_size, row_count)
        sample_df = df.sample(n=n, random_state=42)
        sample_str = sample_df.to_string(index=False)
        sample_data = [
            {str(k): _to_json_safe(v) for k, v in row.items()}
            for row in sample_df.to_dict(orient="records")
        ]

    summary = CsvSummary(
        column_types=column_types,
        sample_rows=sample_str,
        sample_data=sample_data,
        row_count=row_count,
        column_count=column_count,
        missing_values_by_column=missing_values_by_column,
        missing_values_handled=missing_values_handled,
        duplicate_rows_removed=n_duplicates,
        row_count_before_cleaning=row_count_before_cleaning,
        row_count_before_sampling=row_count_before_sampling,
        sampled_rows=sampled_rows,
        column_unique_counts=column_unique_counts,
        column_numeric_stats=column_numeric_stats,
    )

    preprocessed_csv = df.to_csv(index=False)
    return summary, preprocessed_csv


def preprocess_csv(
    csv_content: str,
    sample_size: int | None = None,
) -> CsvSummary:
    """Preprocess CSV content and extract summarized information.

    Steps:
    1. Parse CSV with pandas
    2. Randomly sample large inputs down to ``CSV_MAX_ANALYSIS_ROWS``
    3. Detect missing values, handle with fill_mean for numeric columns
    4. Detect and remove duplicate rows
    5. Extract column types, sample rows, and counts

    Args:
        csv_content: Raw CSV text.
        sample_size: Number of rows to randomly sample. Uses
            settings.CSV_SAMPLE_SIZE if not provided.

    Returns:
        CsvSummary with column types, sample rows, counts, and preprocessing info.
    """
    summary, _ = preprocess_csv_with_data(csv_content, sample_size)
    return summary
