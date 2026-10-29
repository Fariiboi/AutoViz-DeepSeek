"""Deterministic text digest of a generated chart.

AutoViz runs on DeepSeek chat models, which accept text only. The critic and the
annotator therefore never receive the rendered PNG. Instead they reason over the
digest built here:

1. the planned task (chart type, features, title, intended insight),
2. a data profile of the plotted columns (taken from ``CsvSummary``),
3. a static reading of the generated matplotlib/seaborn code — axis labels,
   scales, orientation, grouping/hue, aggregation, plotting calls — plus which
   planned features the code actually references.

Everything is extracted with plain regexes, so the digest is deterministic,
cheap, and works offline. It carries no pixel data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from autoviz.pipeline.csv import CsvSummary

_MAX_CODE_CHARS = 12_000

_QUOTED_ARG = r"""[fFrRbB]?['"]([^'"\n]{1,120})['"]"""

# Matplotlib / seaborn / plotly calls that say nothing about the chart itself.
_PLOT_CALL_NOISE = {
    "savefig",
    "close",
    "show",
    "figure",
    "subplots",
    "style",
    "use",
    "rcParams",
    "rc",
    "legend",
    "grid",
    "title",
    "xlabel",
    "ylabel",
    "xticks",
    "yticks",
    "xlim",
    "ylim",
    "suptitle",
    "tight_layout",
    "colorbar",
    "text",
    "annotate",
    "set",
    "set_theme",
}


def _first_quoted(patterns: tuple[str, ...], code: str) -> str | None:
    """Return the first quoted string matched by any pattern."""
    for pattern in patterns:
        match = re.search(pattern, code)
        if match:
            return match.group(1).strip()
    return None


def _set_axis_labels(code: str) -> tuple[str | None, str | None]:
    """Handle seaborn figure-level ``set_axis_labels("x", "y")``."""
    pattern = r"set_axis_labels\(\s*" + _QUOTED_ARG + r"\s*,\s*" + _QUOTED_ARG
    match = re.search(pattern, code)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


def _unique_params(code: str, name: str, limit: int = 4) -> list[str]:
    """Collect unique quoted values passed as ``name="value"``."""
    pattern = r"\b" + re.escape(name) + r"\s*=\s*" + _QUOTED_ARG
    values: list[str] = []
    for value in re.findall(pattern, code):
        value = value.strip()
        if value and value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def _axis_scale(code: str, axis: str) -> str:
    """Return 'log' when the axis scale is set to logarithmic, else 'linear'."""
    patterns = (
        rf"set_{axis}scale\(\s*['\"]log",
        rf"\.{axis}scale\(\s*['\"]log",
        rf"set\(\s*[^)]*{axis}scale\s*=\s*['\"]log",
    )
    return "log" if any(re.search(p, code) for p in patterns) else "linear"


def _plotting_calls(code: str) -> list[str]:
    """List distinct plotting calls (``sns.barplot``, ``plt.bar``, ...)."""
    calls: list[str] = []
    for module, func in re.findall(r"\b(sns|plt|px)\.(\w+)\(", code):
        if func in _PLOT_CALL_NOISE:
            continue
        name = f"{module}.{func}"
        if name not in calls:
            calls.append(name)
    return calls


def _groupby_columns(code: str) -> list[str]:
    """Extract the column names passed to ``groupby(...)``."""
    columns: list[str] = []
    for group in re.findall(r"groupby\(\s*\[([^\]]{1,200})\]", code):
        columns.extend(re.findall(r"['\"]([^'\"]{1,80})['\"]", group))
    if not columns:
        columns = re.findall(r"groupby\(\s*" + _QUOTED_ARG, code)
    unique: list[str] = []
    for column in columns:
        column = column.strip()
        if column and column not in unique:
            unique.append(column)
    return unique


def _aggregation(code: str) -> list[str]:
    """Describe aggregation / grouping constructs found in the code."""
    parts: list[str] = []
    group_cols = _groupby_columns(code)
    if group_cols:
        parts.append("groupby(" + ", ".join(f"'{c}'" for c in group_cols) + ")")
    for func in ("mean", "median", "sum", "count", "nunique", "value_counts", "pivot_table", "agg"):
        if re.search(rf"\.{func}\(", code):
            parts.append(f".{func}()")
    aggfunc = _unique_params(code, "aggfunc", limit=2)
    if aggfunc:
        parts.append("aggfunc=" + ", ".join(aggfunc))
    return parts


def _orientation(code: str) -> str | None:
    if re.search(r"orient\s*=\s*['\"]h", code) or re.search(r"\bbarh\(", code):
        return "horizontal"
    if re.search(r"orient\s*=\s*['\"]v", code) or re.search(r"\bbar\(", code):
        return "vertical"
    return None


def _fmt_num(value: object) -> str:
    try:
        return f"{float(value):,.4g}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"


def _feature_profile(features: list[str], summary: CsvSummary) -> str:
    """Build per-column evidence the critic can act on."""
    lines: list[str] = [
        f"Dataset: {summary.row_count:,} rows x {summary.column_count:,} columns "
        f"(after cleaning).",
    ]
    for name in features:
        dtype = summary.column_types.get(name)
        if dtype is None:
            lines.append(
                f"- `{name}`: NOT FOUND in the dataset columns — "
                "the planner may have invented it."
            )
            continue
        parts = [f"- `{name}`: {dtype}"]
        unique = summary.column_unique_counts.get(name)
        if unique is not None:
            unit = "value" if unique == 1 else "values"
            parts.append(f"{unique:,} unique {unit}")
            if unique <= 1:
                parts.append("CONSTANT column (no variation to plot)")
        stats = summary.column_numeric_stats.get(name)
        if stats:
            parts.append(f"mean={_fmt_num(stats.get('mean'))}")
            parts.append(f"std={_fmt_num(stats.get('std'))}")
        missing = summary.missing_values_by_column.get(name, 0)
        if missing:
            rate = (missing / summary.row_count * 100) if summary.row_count else 0.0
            parts.append(f"{missing:,} missing ({rate:.1f}%)")
        lines.append(", ".join(parts))
    return "\n".join(lines)


def _rendering_plan(features: list[str], code: str | None) -> str:
    """Summarise what the generated code actually draws."""
    if not code:
        return "- No generated code available."

    label_x, label_y = _set_axis_labels(code)
    label_x = label_x or _first_quoted(
        (r"\.set_xlabel\(\s*" + _QUOTED_ARG, r"\.xlabel\(\s*" + _QUOTED_ARG), code
    )
    label_y = label_y or _first_quoted(
        (r"\.set_ylabel\(\s*" + _QUOTED_ARG, r"\.ylabel\(\s*" + _QUOTED_ARG), code
    )
    title = _first_quoted(
        (
            r"\.set_title\(\s*" + _QUOTED_ARG,
            r"\.suptitle\(\s*" + _QUOTED_ARG,
            r"\.title\(\s*" + _QUOTED_ARG,
        ),
        code,
    )

    lines: list[str] = []
    if label_x or label_y:
        lines.append(f"- Axis labels: x={label_x or 'unlabelled'}, y={label_y or 'unlabelled'}")
    else:
        lines.append("- Axis labels: none set explicitly (axes fall back to column names)")

    lines.append(f"- Scales: x={_axis_scale(code, 'x')}, y={_axis_scale(code, 'y')}")

    orientation = _orientation(code)
    if orientation:
        lines.append(f"- Orientation: {orientation}")

    params: list[str] = []
    for name in ("x", "y", "hue", "size", "style", "col", "row", "bins", "stat", "log_scale"):
        values = _unique_params(code, name)
        if values:
            params.append(f"{name}=" + ", ".join(values))
    if params:
        lines.append("- Plot parameters: " + "; ".join(params))

    aggregation = _aggregation(code)
    if aggregation:
        lines.append("- Aggregation: " + "; ".join(aggregation))

    calls = _plotting_calls(code)
    if calls:
        lines.append("- Plotting calls: " + ", ".join(calls))

    if title:
        lines.append(f"- Rendered title: {title}")

    referenced = [f"`{f}`" for f in features if f in code]
    missing = [f"`{f}`" for f in features if f not in code]
    if referenced:
        lines.append("- Planned features used in code: " + ", ".join(referenced))
    if missing:
        lines.append("- Planned features NOT referenced in code: " + ", ".join(missing))

    return "\n".join(lines)


@dataclass(frozen=True)
class ChartDigest:
    """Text-only description of one chart, ready to embed in a prompt."""

    task: str
    data: str
    rendering: str
    code: str | None = None

    def to_prompt_text(self) -> str:
        """Render the digest as a single prompt block."""
        sections = [
            "## 1. Chart task",
            self.task,
            "",
            "## 2. Data profile of plotted columns",
            self.data,
            "",
            "## 3. Rendering plan inferred from the generated code",
            self.rendering,
        ]
        if self.code:
            code = self.code
            note = ""
            if len(code) > _MAX_CODE_CHARS:
                code = code[:_MAX_CODE_CHARS]
                note = "\n# ... code truncated ..."
            sections.extend(
                [
                    "",
                    "## 4. Generated code (reference)",
                    "```python",
                    code + note,
                    "```",
                ]
            )
        return "\n".join(sections)


def build_chart_digest(
    *,
    chart_type: str,
    features: list[str],
    title: str,
    description: str | None,
    csv_summary: CsvSummary,
    chart_code: str | None = None,
) -> ChartDigest:
    """Assemble the text digest used by the text-only critic and annotator.

    Args:
        chart_type: Planned chart type, e.g. "bar" or "heatmap".
        features: Columns the planner asked for.
        title: Planned chart title.
        description: One-sentence intended insight (may be None).
        csv_summary: Dataset summary providing the column profile.
        chart_code: Code that produced the chart; parsed for axes/scales/etc.

    Returns:
        A :class:`ChartDigest` whose ``to_prompt_text()`` is prompt-ready.
    """
    feature_list = ", ".join(features) if features else "(none)"
    task_lines = [
        f"- Chart type: {chart_type}",
        f"- Planned features: {feature_list}",
        f"- Title: {title}",
    ]
    if description:
        task_lines.append(f"- Intended insight: {description}")

    return ChartDigest(
        task="\n".join(task_lines),
        data=_feature_profile(features, csv_summary),
        rendering=_rendering_plan(features, chart_code),
        code=chart_code,
    )
