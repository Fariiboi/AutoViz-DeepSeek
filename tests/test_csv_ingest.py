"""Tests for CSV encoding detection and large-file sampling."""

from __future__ import annotations

import pandas as pd

from autoviz.config import settings
from autoviz.pipeline.csv import preprocess_csv_with_data, read_text_auto


def test_read_text_auto_utf8(tmp_path):
    path = tmp_path / "utf8.csv"
    path.write_bytes("城市,销量\n北京,1\n".encode())

    assert read_text_auto(path) == "城市,销量\n北京,1\n"


def test_read_text_auto_utf8_with_bom(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes("\ufeffa,b\n1,2\n".encode())

    assert read_text_auto(path) == "a,b\n1,2\n"


def test_read_text_auto_gbk_fallback(tmp_path):
    text = "城市,销量\n上海,2\n"
    path = tmp_path / "gbk.csv"
    path.write_bytes(text.encode("gbk"))

    assert read_text_auto(path) == text


def build_csv(rows: int) -> str:
    return pd.DataFrame(
        {"x": range(rows), "y": [f"cat{index % 3}" for index in range(rows)]}
    ).to_csv(index=False)


def test_large_csv_is_sampled():
    summary, cleaned = preprocess_csv_with_data(build_csv(2500))

    assert summary.row_count_before_sampling == 2500
    assert summary.sampled_rows == settings.CSV_MAX_ANALYSIS_ROWS
    assert summary.row_count == settings.CSV_MAX_ANALYSIS_ROWS
    assert summary.row_count_before_cleaning == settings.CSV_MAX_ANALYSIS_ROWS
    # header + sampled rows, nothing more
    assert cleaned.count("\n") == settings.CSV_MAX_ANALYSIS_ROWS + 1
    assert "random sample of 2,500 rows" in summary.to_prompt_text()


def test_small_csv_is_not_sampled():
    summary, cleaned = preprocess_csv_with_data(build_csv(50))

    assert summary.sampled_rows == 0
    assert summary.row_count == summary.row_count_before_sampling == 50
    assert cleaned.count("\n") == 51
    assert "random sample" not in summary.to_prompt_text()


def test_sampling_is_deterministic():
    csv = build_csv(2200)

    _, first = preprocess_csv_with_data(csv)
    _, second = preprocess_csv_with_data(csv)

    assert first == second


def test_roundtrip_dict_keeps_sampling_info():
    summary, _ = preprocess_csv_with_data(build_csv(2500))

    restored = type(summary).from_dict(summary.to_dict())

    assert restored.row_count_before_sampling == 2500
    assert restored.sampled_rows == settings.CSV_MAX_ANALYSIS_ROWS
