"""Bảng thống kê sau mỗi lần chạy: số câu pass / review / fail ở từng bước (review P5).

API: ``vietvoice.data.stats.RunStats``.
"""

import json

import pytest

from vietvoice.data.stats import RunStats


def test_counts_per_step_in_insertion_order():
    stats = RunStats()
    stats.add_many("format", ["pass", "pass", "fail"])
    stats.add("vad", "review")
    stats.add("format", "review")
    assert stats.to_dict() == {
        "format": {"pass": 2, "review": 1, "fail": 1, "total": 4},
        "vad": {"pass": 0, "review": 1, "fail": 0, "total": 1},
    }


def test_unknown_status_rejected():
    with pytest.raises(ValueError, match="trạng thái"):
        RunStats().add("format", "ok")


def test_table_is_vietnamese_and_lists_steps():
    stats = RunStats()
    stats.add_many("format", ["pass", "fail"])
    table = stats.table()
    assert "Bước" in table and "format" in table and "Tổng" in table


def test_write_json_is_reproducible(tmp_path):
    stats = RunStats()
    stats.add_many("loudness", ["pass", "review"])
    stats.write_json(tmp_path / "a.json")
    stats.write_json(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["loudness"]["review"] == 1
