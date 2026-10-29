"""Visualization task selector.

Runs between viz_main_agent (planning) and SubAgentManager (execution).
Pure Python rules + heuristics — no LLM calls.

For each VizTaskSpec produced by the planner, the selector:
  1. Checks feasibility (column existence, type compatibility, missing rate, cardinality).
  2. Scores information gain (missing rate penalty, cardinality bonus, numeric variance,
     sample variety, within-batch redundancy penalty).
  3. Emits decision: "accept" | "reject" | "revise".
     - revise: returns an amended VizTaskSpec (different chart_type / trimmed features).
     - reject: task is dropped; caller should emit a skipped VizResult with skip_kind="selector".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from autoviz.agents.planner import VizTaskSpec
from autoviz.pipeline.csv import CsvSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

_NUMERIC_TYPES = {"integer", "float"}
_CATEGORICAL_TYPES = {"string", "boolean"}
_ORDERED_TYPES = {"integer", "float", "datetime"}


def _is_numeric(col_type: str) -> bool:
    return col_type in _NUMERIC_TYPES


def _is_categorical(col_type: str) -> bool:
    return col_type in _CATEGORICAL_TYPES


def _is_ordered(col_type: str) -> bool:
    return col_type in _ORDERED_TYPES


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class SelectorDecision:
    """Result of evaluating a single VizTaskSpec."""

    task: VizTaskSpec
    decision: Literal["accept", "reject", "revise"]
    reason_code: str  # e.g. "accepted", "type_mismatch", "high_missing", "low_ig", "constant_col"
    ig_score: float  # 0.0 - 1.0
    revised_task: VizTaskSpec | None = None  # populated only when decision == "revise"


# ---------------------------------------------------------------------------
# Feasibility rules
# ---------------------------------------------------------------------------


def _missing_rate(col: str, csv_summary: CsvSummary) -> float:
    """Return the fraction of missing values for *col* (0.0 - 1.0)."""
    total = csv_summary.row_count
    if total <= 0:
        return 0.0
    missing = csv_summary.missing_values_by_column.get(col, 0)
    return missing / total


def _check_feasibility(
    task: VizTaskSpec,
    csv_summary: CsvSummary,
    max_missing_rate: float,
) -> tuple[bool, str, VizTaskSpec | None]:
    """Check feasibility of a viz task.

    Returns:
        (ok, reason_code, revised_task)
        - ok=True  → feasible (reason_code="ok", revised_task=None)
        - ok=False → infeasible (reason_code explains why, revised_task may offer an alternative)
    """
    col_types = csv_summary.column_types
    unique_counts = csv_summary.column_unique_counts
    chart = task.chart_type
    features = task.features

    # --- 1. Row count floor ---
    if csv_summary.row_count < 5:
        return False, "insufficient_rows", None

    # --- 2. All features must exist ---
    missing_cols = [f for f in features if f not in col_types]
    if missing_cols:
        return False, f"unknown_columns:{','.join(missing_cols)}", None

    # --- 3. Constant column check (unique_count <= 1) ---
    constant_cols = [
        f for f in features if unique_counts.get(f, 2) <= 1
    ]
    if constant_cols:
        return False, f"constant_col:{','.join(constant_cols)}", None

    # --- 4. Missing rate check on primary features ---
    high_missing = [
        f for f in features if _missing_rate(f, csv_summary) > max_missing_rate
    ]
    if high_missing:
        return False, f"high_missing:{','.join(high_missing)}", None

    # --- 5. Type x chart compatibility (with revise where possible) ---
    types = [col_types[f] for f in features]

    # scatter / bubble / kde / scatter_matrix — need ≥2 numeric features
    if chart in {"scatter", "bubble", "kde", "scatter_matrix"}:
        numeric_features = [f for f in features if _is_numeric(col_types[f])]
        min_numeric = 3 if chart == "scatter_matrix" else 2
        if len(numeric_features) < min_numeric:
            if chart in {"scatter", "bubble"} and len(numeric_features) == 1 and len(features) > 1:
                # Try dropping non-numeric columns
                trimmed = [f for f in features if _is_numeric(col_types[f])]
                # Need exactly 2 for scatter
                if len(trimmed) >= 2:
                    revised = VizTaskSpec(
                        chart_type=chart,
                        features=trimmed[:2],
                        title=task.title,
                        description=task.description,
                    )
                    return False, "type_mismatch:non_numeric_dropped", revised
            return False, f"type_mismatch:{chart}_needs_{min_numeric}_numeric", None

    # histogram / box / violin — need ≥1 numeric feature
    elif chart in {"histogram", "box", "violin"}:
        if not any(_is_numeric(t) for t in types):
            return False, f"type_mismatch:{chart}_needs_numeric", None

    # line / area — need ordered x-axis (datetime preferred); revise to bar if no datetime
    elif chart in {"line", "area"}:
        has_datetime = any(col_types[f] == "datetime" for f in features)
        has_numeric_y = any(_is_numeric(col_types[f]) for f in features)
        if not has_numeric_y:
            return False, "type_mismatch:line_needs_numeric_y", None
        if not has_datetime:
            # Revise: line without a time axis is usually better as bar
            revised = VizTaskSpec(
                chart_type="bar",
                features=features,
                title=task.title,
                description=task.description,
            )
            return False, "revised:line_no_datetime->bar", revised

    # heatmap — needs ≥2 numeric cols (correlation mode) or 2 categorical + 1 numeric
    elif chart == "heatmap":
        numeric_features = [f for f in features if _is_numeric(col_types[f])]
        cat_features = [f for f in features if _is_categorical(col_types[f])]
        if len(numeric_features) < 2 and not (len(cat_features) >= 2 and len(numeric_features) >= 1):
            return False, "type_mismatch:heatmap_needs_2numeric_or_2cat1num", None

    # bar / pie / funnel / treemap — need ≥1 categorical feature
    elif chart in {"bar", "pie", "funnel", "treemap"}:
        if not any(_is_categorical(t) for t in types):
            return False, f"type_mismatch:{chart}_needs_categorical", None
        # pie with too many categories is meaningless
        if chart == "pie":
            cat_col = next((f for f in features if _is_categorical(col_types[f])), None)
            if cat_col and unique_counts.get(cat_col, 0) > 20:
                return False, "pie_too_many_categories", None

    # stacked_bar / waterfall — need 1 categorical + 1 numeric
    elif chart in {"stacked_bar", "waterfall"}:
        has_cat = any(_is_categorical(t) for t in types)
        has_num = any(_is_numeric(t) for t in types)
        if not (has_cat and has_num):
            return False, f"type_mismatch:{chart}_needs_cat_and_numeric", None

    return True, "ok", None


# ---------------------------------------------------------------------------
# Information gain scoring
# ---------------------------------------------------------------------------

# Columns appearing in this many or fewer tasks get full redundancy score.
# Raised from 3 → 5: key grouping columns (region, category, status) legitimately
# appear across many chart types and should not be penalised early.
_MAX_REDUNDANCY_APPEARANCES = 5


def _build_col_appearances(tasks: list[VizTaskSpec]) -> dict[str, int]:
    """Count how many tasks each column appears in across the full batch."""
    appearances: dict[str, int] = {}
    for t in tasks:
        for f in t.features:
            appearances[f] = appearances.get(f, 0) + 1
    return appearances


def _score_ig(
    task: VizTaskSpec,
    csv_summary: CsvSummary,
    col_appearances: dict[str, int],
) -> float:
    """Heuristic information gain score in [0, 1].

    Component weights sum to 1.0:
      missing_penalty   : 0.40
      cardinality_bonus : 0.30
      variance_bonus    : 0.20
      redundancy_penalty: 0.10

    Multi-feature bonus: +0.05 for tasks with ≥3 columns (cross-variable insight).
    """
    col_types = csv_summary.column_types
    unique_counts = csv_summary.column_unique_counts
    numeric_stats = csv_summary.column_numeric_stats
    row_count = max(csv_summary.row_count, 1)
    features = task.features

    # --- Component 1: missing rate penalty (weight 0.40) ---
    avg_missing = sum(_missing_rate(f, csv_summary) for f in features) / max(len(features), 1)
    missing_score = (1.0 - avg_missing) * 0.40

    # --- Component 2: cardinality bonus (weight 0.30) ---
    # For categorical columns the scoring distinguishes three zones:
    #   - ≤1 unique: constant, score 0  (infeasible, already caught by feasibility check)
    #   - 2-20 unique: ideal grouping variable, score 0.85  ← was incorrectly penalised before
    #   - 21-50 unique: moderate, score 0.50
    #   - >50% of rows unique: ID-like, score 0.15
    # Numeric / datetime columns are assumed meaningful (score 1.0).
    cardinality_scores: list[float] = []
    for f in features:
        ft = col_types.get(f, "string")
        if _is_categorical(ft):
            uc = unique_counts.get(f, 0)
            if uc <= 1:
                card_score = 0.0
            elif uc <= 20:
                # Sweet spot for grouping / comparison charts (e.g. gender, region, status)
                card_score = 0.85
            elif uc <= 50:
                card_score = 0.50
            else:
                # High-cardinality categorical: penalise but keep a floor
                ratio = uc / row_count
                card_score = max(1.0 - ratio, 0.10)
            cardinality_scores.append(card_score)
        else:
            # Numeric / datetime: meaningful unless constant (caught earlier)
            cardinality_scores.append(1.0)
    cardinality_score = (sum(cardinality_scores) / max(len(cardinality_scores), 1)) * 0.30

    # --- Component 3: numeric variance bonus (weight 0.20) ---
    variance_scores: list[float] = []
    for f in features:
        ft = col_types.get(f, "string")
        if _is_numeric(ft):
            stats = numeric_stats.get(f, {})
            std = float(stats.get("std", 0.0))
            mean = float(stats.get("mean", 0.0))
            if _is_nan(std) or _is_nan(mean):
                variance_scores.append(0.0)
            elif mean != 0:
                # Coefficient of variation proxy; cap at 1
                cv = abs(std / mean)
                variance_scores.append(min(cv, 1.0))
            elif std > 0:
                variance_scores.append(1.0)
            else:
                variance_scores.append(0.0)
        else:
            variance_scores.append(0.75)  # Non-numeric assumed varied
    variance_score = (sum(variance_scores) / max(len(variance_scores), 1)) * 0.20

    # --- Component 4: within-batch redundancy penalty (weight 0.10) ---
    # Softer than before: penalty only kicks in after _MAX_REDUNDANCY_APPEARANCES,
    # and decays at 0.7^n instead of 0.5^n so common columns are not killed off.
    max_appearances = max(col_appearances.get(f, 1) for f in features)
    if max_appearances <= _MAX_REDUNDANCY_APPEARANCES:
        redundancy_score = 0.10
    else:
        redundancy_score = 0.10 * (0.7 ** (max_appearances - _MAX_REDUNDANCY_APPEARANCES))

    total = missing_score + cardinality_score + variance_score + redundancy_score

    # --- Multi-feature bonus ---
    # 3-variable charts (e.g. scatter with colour, grouped bar, heatmap, bubble) reveal
    # conditional patterns that bivariate charts miss. Give them a small lift so they are
    # not unfairly penalised by the redundancy component touching multiple columns.
    if len(features) >= 3:
        total += 0.05

    return round(min(max(total, 0.0), 1.0), 4)


def _is_nan(v: object) -> bool:
    try:
        import math
        return math.isnan(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def select_tasks(
    tasks: list[VizTaskSpec],
    csv_summary: CsvSummary,
    *,
    min_ig_score: float = 0.2,
    max_missing_rate: float = 0.7,
) -> list[SelectorDecision]:
    """Evaluate each VizTaskSpec and return a SelectorDecision for every task.

    Args:
        tasks: List of tasks from viz_main_agent.
        csv_summary: Structured CSV summary (must include column_unique_counts,
                     column_numeric_stats; falls back gracefully if absent).
        min_ig_score: Tasks scoring below this threshold are rejected.
        max_missing_rate: Primary feature missing rate ceiling (0-1).

    Returns:
        One SelectorDecision per input task, preserving order.
    """
    col_appearances = _build_col_appearances(tasks)
    decisions: list[SelectorDecision] = []

    for task in tasks:
        feasible, reason, revised_spec = _check_feasibility(
            task, csv_summary, max_missing_rate
        )

        if not feasible:
            if revised_spec is not None:
                # Revise: re-score the revised spec
                ig = _score_ig(revised_spec, csv_summary, col_appearances)
                if ig < min_ig_score:
                    decisions.append(
                        SelectorDecision(
                            task=task,
                            decision="reject",
                            reason_code=f"low_ig_after_revise:{reason}",
                            ig_score=ig,
                        )
                    )
                else:
                    decisions.append(
                        SelectorDecision(
                            task=task,
                            decision="revise",
                            reason_code=reason,
                            ig_score=ig,
                            revised_task=revised_spec,
                        )
                    )
                    logger.debug(
                        "Selector REVISE %s %s -> %s %s (reason=%s ig=%.2f)",
                        task.chart_type, task.features,
                        revised_spec.chart_type, revised_spec.features,
                        reason, ig,
                    )
            else:
                decisions.append(
                    SelectorDecision(
                        task=task,
                        decision="reject",
                        reason_code=reason,
                        ig_score=0.0,
                    )
                )
                logger.debug(
                    "Selector REJECT %s %s (reason=%s)",
                    task.chart_type, task.features, reason,
                )
            continue

        ig = _score_ig(task, csv_summary, col_appearances)

        if ig < min_ig_score:
            decisions.append(
                SelectorDecision(
                    task=task,
                    decision="reject",
                    reason_code="low_ig",
                    ig_score=ig,
                )
            )
            logger.debug(
                "Selector REJECT %s %s (low_ig=%.2f)",
                task.chart_type, task.features, ig,
            )
        else:
            decisions.append(
                SelectorDecision(
                    task=task,
                    decision="accept",
                    reason_code="accepted",
                    ig_score=ig,
                )
            )
            logger.debug(
                "Selector ACCEPT %s %s (ig=%.2f)",
                task.chart_type, task.features, ig,
            )

    accepted = sum(1 for d in decisions if d.decision == "accept")
    revised = sum(1 for d in decisions if d.decision == "revise")
    rejected = sum(1 for d in decisions if d.decision == "reject")
    logger.info(
        "Selector: %d accepted, %d revised, %d rejected (from %d tasks)",
        accepted, revised, rejected, len(tasks),
    )

    return decisions
