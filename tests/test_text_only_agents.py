"""Regression tests for the DeepSeek text-only refactor.

These tests never call the network: they cover configuration defaults, the
deterministic chart digest, and the fact that the critic/annotator are wired to
DeepSeek and never emit image content.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import get_args

from pydantic_ai.models.test import TestModel

from autoviz.agents import annotator, critic
from autoviz.agents.chart_digest import build_chart_digest
from autoviz.config import Settings, settings
from autoviz.pipeline.csv import CsvSummary

_ANALYSIS_CODE = '''import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/home/user/input.csv")
agg = df.groupby(["cat"])["val"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=agg, x="val", y="cat", hue="grp", orient="h", ax=ax)
ax.set_xlabel("Value")
ax.set_ylabel("Category")
ax.set_xscale("log")
ax.set_title("Value by category")
plt.savefig("/home/user/output.png", dpi=150, bbox_inches="tight")
'''


def _summary() -> CsvSummary:
    return CsvSummary(
        column_types={"cat": "string", "val": "float", "grp": "string"},
        row_count=10,
        column_count=3,
        column_unique_counts={"cat": 3, "val": 10, "grp": 2},
        column_numeric_stats={"val": {"mean": 1.5, "std": 0.5}},
    )


def test_settings_defaults_are_deepseek():
    fields = Settings.model_fields

    assert fields["AI_MODEL"].default == "deepseek-chat"
    assert fields["DEEPSEEK_BASE_URL"].default == "https://api.deepseek.com/v1"
    assert "DEEPSEEK_API_KEY" in fields
    assert "OPENROUTER_API_KEY" not in fields
    assert "OPENROUTER_BASE_URL" not in fields


def test_agents_are_wired_to_deepseek():
    for agent in (critic._get_critic_agent(), annotator._get_annotation_agent()):
        assert agent.model.model_name == settings.AI_MODEL
        assert str(agent.model.client.base_url).rstrip("/") == "https://api.deepseek.com/v1"


def test_agents_send_no_image_content():
    for module in (annotator, critic):
        source = inspect.getsource(module)
        assert "BinaryContent" not in source
        assert "media_type" not in source


def test_critic_schema_drops_vision_only_failures():
    assert "empty_plot" not in get_args(critic.VizFailureType)
    assert "wrong_encoding" not in get_args(critic.VizFailureType)
    assert "weak_signal" in get_args(critic.VizFailureType)


def test_digest_describes_axes_scale_orientation_and_aggregation():
    digest = build_chart_digest(
        chart_type="bar",
        features=["cat", "val", "grp"],
        title="Value by category",
        description="Average value per category and group.",
        csv_summary=_summary(),
        chart_code=_ANALYSIS_CODE,
    )
    text = digest.to_prompt_text()

    assert "Axis labels: x=Value, y=Category" in text
    assert "Scales: x=log, y=linear" in text
    assert "Orientation: horizontal" in text
    assert "groupby('cat')" in text
    assert "Plotting calls: sns.barplot" in text
    assert "Planned features used in code: `cat`, `val`, `grp`" in text
    assert "df.groupby" in text  # code excerpt included for reference


def test_digest_flags_constant_missing_and_unreferenced_features():
    summary = CsvSummary(
        column_types={"flag": "integer"},
        row_count=100,
        column_count=1,
        column_unique_counts={"flag": 1},
        column_numeric_stats={"flag": {"mean": 3.0, "std": 0.0}},
        missing_values_by_column={"ghost": 95},
    )
    digest = build_chart_digest(
        chart_type="histogram",
        features=["flag", "ghost", "invented"],
        title="Flag distribution",
        description=None,
        csv_summary=summary,
        chart_code="plt.hist(df['flag'])",
    )
    text = digest.to_prompt_text()

    assert "CONSTANT column" in text
    assert "1 unique value" in text
    assert "NOT FOUND in the dataset columns" in text
    assert "NOT referenced in code: `ghost`, `invented`" in text


def test_digest_without_code_still_works():
    digest = build_chart_digest(
        chart_type="pie",
        features=["cat"],
        title="Category share",
        description=None,
        csv_summary=_summary(),
        chart_code=None,
    )
    text = digest.to_prompt_text()

    assert "No generated code available." in text
    assert "## 4. Generated code" not in text


def test_critic_accepts_text_digest_only(monkeypatch):
    """End-to-end prompt/structured-output path, no network, no image argument."""
    monkeypatch.setattr(critic, "_critic_agent", critic._build_critic_agent())
    with critic._critic_agent.override(model=TestModel()):
        verdict = asyncio.run(
            critic.critique_chart(
                chart_type="bar",
                features=["cat", "val"],
                title="Value by category",
                description="Compare values.",
                csv_summary=_summary(),
                chart_code=_ANALYSIS_CODE,
            )
        )

    assert isinstance(verdict, critic.VizCritique)
    assert verdict.action in get_args(critic.VizCriticAction)


def test_annotator_accepts_text_digest_only(monkeypatch):
    monkeypatch.setattr(annotator, "_annotation_agent", annotator._build_annotation_agent())
    with annotator._annotation_agent.override(model=TestModel()):
        text = asyncio.run(
            annotator.generate_chart_annotation(
                chart_type="bar",
                features=["cat", "val"],
                title="Value by category",
                csv_summary=_summary(),
                description="Compare values.",
                chart_code=_ANALYSIS_CODE,
            )
        )

    assert isinstance(text, str)
    assert text.strip()
