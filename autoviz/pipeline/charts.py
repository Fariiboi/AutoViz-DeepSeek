"""Visualization pipeline.

Orchestrates the full multi-agent visualization flow with optional multi-turn:

  For each turn (1..VIZ_TURNS):
    1. Main agent plans VIZ_TARGET_CHARTS visualization tasks.
    2. SubAgentManager runs tasks concurrently: code → sandbox → critic → (retry or semantic skip) → annotation if accepted.
    3. Semantic critic rejections trigger replacement waves (bounded); failures feed the next plan call.
    4. Turn results are accumulated; next turn excludes already-generated charts and prior critic failures.

  Planned baseline = VIZ_TARGET_CHARTS * VIZ_TURNS; actual task count may be higher when replacements run.

Usage:
    results = await run_visualization(
        enhanced_csv=enhanced_csv_str,
        csv_summary=csv_summary,
        on_event=lambda result: ...,
        on_turn_start=lambda turn, total: ...,
    )
"""

import logging
from collections.abc import Callable
from pathlib import Path

from autoviz.agents.planner import VizTaskSpec, plan_visualizations
from autoviz.agents.selector import (  # SelectorDecision used in type hint
    SelectorDecision,
    select_tasks,
)
from autoviz.config import settings
from autoviz.events import MessageBus, VizResult, VizTask
from autoviz.pipeline.csv import CsvSummary
from autoviz.pipeline.runner import HistoryManager, SubAgentManager, make_task_id

logger = logging.getLogger(__name__)


def _specs_to_tasks(specs: list[VizTaskSpec], *, replan_generation: int = 0) -> list[VizTask]:
    """Convert VizTaskSpec objects from the main agent into VizTask objects."""
    return [
        VizTask(
            task_id=make_task_id(),
            chart_type=spec.chart_type,
            features=spec.features,
            title=spec.title,
            description=spec.description,
            replan_generation=replan_generation,
        )
        for spec in specs
    ]


def _collect_exclude_tasks(results: list[VizResult]) -> list[tuple[str, list[str]]]:
    """Collect (chart_type, features) from results for exclusion in next turn."""
    return [(r.chart_type, r.features) for r in results]


def _failure_dict_from_viz_result(r: VizResult) -> dict:
    """Shape for planner prior_critic_failures."""
    md = r.metadata or {}
    return {
        "chart_type": r.chart_type,
        "features": list(r.features),
        "failure_type": md.get("failure_type", ""),
        "critic_reason": md.get("critic_reason", ""),
        "action": md.get("action", ""),
    }


def _apply_selector(
    specs: list[VizTaskSpec],
    csv_summary: CsvSummary,
    on_event: Callable[[VizResult], None] | None,
) -> tuple[list[VizTaskSpec], list[VizResult]]:
    """Run the Selector layer over a list of planned specs.

    Returns:
        (accepted_specs, selector_skipped_results)
        - accepted_specs: specs to forward to SubAgentManager (accept + revise).
        - selector_skipped_results: VizResult(skipped) for every rejected spec,
          ready to be reported via on_event and accumulated in all_results.
    """
    decisions: list[SelectorDecision] = select_tasks(
        specs,
        csv_summary,
        min_ig_score=settings.VIZ_SELECTOR_MIN_IG_SCORE,
        max_missing_rate=settings.VIZ_SELECTOR_MAX_MISSING_RATE,
    )

    accepted_specs: list[VizTaskSpec] = []
    skipped_results: list[VizResult] = []

    for decision in decisions:
        if decision.decision == "accept":
            accepted_specs.append(decision.task)
        elif decision.decision == "revise" and decision.revised_task is not None:
            accepted_specs.append(decision.revised_task)
        else:
            # Emit a skipped VizResult so the frontend can surface it via VizPipelineIssue
            result = VizResult(
                task_id=make_task_id(),
                chart_type=decision.task.chart_type,
                features=decision.task.features,
                status="skipped",
                metadata={
                    "skip_kind": "selector",
                    "critic_reason": decision.reason_code,
                    "ig_score": decision.ig_score,
                    "title": decision.task.title,
                },
            )
            skipped_results.append(result)
            if on_event:
                on_event(result)

    return accepted_specs, skipped_results


async def _run_turn_with_critic_replacements(
    csv_summary: CsvSummary,
    manager: SubAgentManager,
    charts_per_turn: int,
    exclude_already: list[tuple[str, list[str]]],
    session_critic_failures: list[dict],
    on_event: Callable[[VizResult], None] | None,
    user_preferences: str | None = None,
) -> list[VizResult]:
    """Plan initial wave, run tasks, plan replacements for semantic critic skips until stable."""
    specs = await plan_visualizations(
        csv_summary,
        exclude_already=exclude_already,
        target_charts=charts_per_turn,
        prior_critic_failures=session_critic_failures or None,
        user_preferences=user_preferences,
    )

    if settings.VIZ_SELECTOR_ENABLED:
        specs, selector_skipped = _apply_selector(specs, csv_summary, on_event)
    else:
        selector_skipped = []

    wave_tasks = _specs_to_tasks(specs, replan_generation=0)
    turn_results: list[VizResult] = list(selector_skipped)
    turn_exclude = list(exclude_already)
    wave_prior_failures: list[dict] = list(session_critic_failures)

    while wave_tasks:
        results = await manager.run_all(wave_tasks, on_progress=on_event)
        turn_results.extend(results)
        turn_exclude.extend(_collect_exclude_tasks(results))

        replan_generations: list[int] = []
        for r in results:
            md = r.metadata or {}
            if r.status == "skipped" and md.get("replacement_requested"):
                wave_prior_failures.append(_failure_dict_from_viz_result(r))
                replan_generations.append(int(md.get("replan_generation", 0)) + 1)

        n_rep = len(replan_generations)
        if n_rep == 0:
            break

        replace_specs = await plan_visualizations(
            csv_summary,
            exclude_already=turn_exclude,
            target_charts=n_rep,
            prior_critic_failures=wave_prior_failures or None,
            user_preferences=user_preferences,
        )
        if not replace_specs:
            logger.warning(
                "Planner returned no replacement tasks for %d critic semantic skip(s)",
                n_rep,
            )
            break

        if settings.VIZ_SELECTOR_ENABLED:
            replace_specs, rep_selector_skipped = _apply_selector(replace_specs, csv_summary, on_event)
            turn_results.extend(rep_selector_skipped)
        else:
            rep_selector_skipped = []

        wave_tasks = []
        for i, spec in enumerate(replace_specs):
            gen = replan_generations[i] if i < len(replan_generations) else replan_generations[-1]
            wave_tasks.extend(_specs_to_tasks([spec], replan_generation=gen))

    return turn_results


async def run_visualization(
    enhanced_csv: str,
    csv_summary: CsvSummary,
    *,
    on_event: Callable[[VizResult], None] | None = None,
    on_turn_start: Callable[[int, int], None] | None = None,
    num_turns: int | None = None,
    target_charts: int | None = None,
    initial_exclude: list[tuple[str, list[str]]] | None = None,
    user_preferences: str | None = None,
    extra_metadata: dict | None = None,
    viz_workspace_override: Path | None = None,
) -> list[VizResult]:
    """Run the full visualization multi-agent pipeline with multi-turn support.

    Args:
        enhanced_csv:  Feature-engineered CSV string from the feature engineering pipeline.
        csv_summary:   CsvSummary of the enhanced CSV (used for LLM context).
        on_event:      Optional callback invoked each time a sub-agent completes a task.
        on_turn_start: Optional callback invoked when a new turn begins (turn: 1-based, total: VIZ_TURNS).
        num_turns:     Optional override for number of planning turns (default: settings.VIZ_TURNS).
        target_charts: Optional override for charts per turn (default: settings.VIZ_TARGET_CHARTS).
        initial_exclude: Optional (chart_type, features) to exclude from the start.
        user_preferences: Optional formatted Q&A context from the MCQ phase.
        viz_workspace_override: When set, PNGs are saved here instead of settings.VIZ_WORKSPACE_DIR.

    Returns:
        List of VizResult, one per planned task across all turns.
    """
    viz_dir = viz_workspace_override if viz_workspace_override is not None else Path(settings.VIZ_WORKSPACE_DIR)
    history_file = (
        viz_workspace_override.parent / "history.md"
        if viz_workspace_override is not None
        else Path(settings.VIZ_HISTORY_FILE)
    )
    turns = num_turns if num_turns is not None else settings.VIZ_TURNS
    charts_per_turn = target_charts if target_charts is not None else settings.VIZ_TARGET_CHARTS

    viz_dir.mkdir(parents=True, exist_ok=True)
    history_file.parent.mkdir(parents=True, exist_ok=True)

    bus = MessageBus()
    history_manager = HistoryManager(history_file)
    manager = SubAgentManager(
        bus=bus,
        history_manager=history_manager,
        enhanced_csv=enhanced_csv,
        csv_summary=csv_summary,
        viz_workspace=viz_dir,
        extra_metadata=extra_metadata,
    )

    all_results: list[VizResult] = []
    exclude_already: list[tuple[str, list[str]]] = list(initial_exclude) if initial_exclude else []
    session_critic_failures: list[dict] = []

    for turn in range(1, turns + 1):
        if on_turn_start:
            on_turn_start(turn, turns)

        results = await _run_turn_with_critic_replacements(
            csv_summary,
            manager,
            charts_per_turn,
            exclude_already,
            session_critic_failures,
            on_event,
            user_preferences=user_preferences,
        )
        all_results.extend(results)
        exclude_already.extend(_collect_exclude_tasks(results))
        for r in results:
            md = r.metadata or {}
            if md.get("skip_kind") == "critic_semantic":
                session_critic_failures.append(_failure_dict_from_viz_result(r))

    return all_results
