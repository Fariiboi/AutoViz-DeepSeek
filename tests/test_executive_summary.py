"""Tests for the executive-summary prompt (offline, no API calls)."""

from __future__ import annotations

from autoviz.agents.summarizer import build_executive_summary_prompt


def test_prompt_includes_report_charts_and_requirements():
    prompt = build_executive_summary_prompt(
        "# 报告\n销售额同比增长 12%",
        [
            {
                "title": "Price vs Age",
                "chart_type": "scatter",
                "features": ["price", "age"],
                "annotation": "价格与年龄呈弱相关",
            }
        ],
        {"row_count": 2000, "column_count": 12, "sampled_rows": 2000},
    )

    assert "历史分析报告" in prompt
    assert "销售额同比增长 12%" in prompt
    assert "Price vs Age" in prompt
    assert "price, age" in prompt
    assert "价格与年龄呈弱相关" in prompt
    assert "450-600 字" in prompt
    assert "恰好 3 条" in prompt
    assert "恰好 2 条" in prompt
    assert "数据行数" in prompt
    assert "2,000" in prompt


def test_prompt_handles_missing_charts_and_dataset():
    prompt = build_executive_summary_prompt("", [])

    assert "本次分析没有生成报告" in prompt
    assert "没有可用的图表元数据" in prompt
    assert "数据集信息" not in prompt


def test_long_report_is_truncated():
    prompt = build_executive_summary_prompt("x" * 25_000, [])

    assert "报告过长" in prompt
    assert "x" * 20_001 not in prompt


def test_chart_limit_and_annotation_truncation():
    charts = [
        {
            "title": f"chart-{index}",
            "chart_type": "bar",
            "features": ["f"],
            "annotation": "a" * 2_000,
        }
        for index in range(30)
    ]

    prompt = build_executive_summary_prompt("report", charts)

    assert "仅展示前 20 张" in prompt
    assert "chart-19" in prompt
    assert "chart-20" not in prompt
    assert "……" in prompt
