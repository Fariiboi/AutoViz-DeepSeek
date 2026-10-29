"""Tests for the Chinese report translation prompt (offline)."""

from __future__ import annotations

from autoviz.agents.translator import MAX_REPORT_CHARS, build_translation_prompt


def test_translation_prompt_keeps_content_and_references():
    report = (
        "# AutoViz Analysis Report\n\n"
        "## Key Findings\n\n"
        "- Average price is 1234.5 (see Visualization 2).\n"
    )

    prompt = build_translation_prompt(report)

    assert "Simplified Chinese" in prompt
    assert "Translate the following English Markdown report" in prompt
    assert "# AutoViz Analysis Report" in prompt
    assert "1234.5" in prompt
    assert "Visualization 2" in prompt


def test_translation_prompt_truncates_long_reports():
    prompt = build_translation_prompt("x" * (MAX_REPORT_CHARS + 100))

    assert "was truncated" in prompt
    assert "x" * (MAX_REPORT_CHARS + 1) not in prompt
