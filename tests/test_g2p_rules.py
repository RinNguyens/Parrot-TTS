"""Luật biến âm theo vùng, đọc từ tests/data/g2p_rules.tsv.

Mỗi dòng trong file TSV kiểm tra một trường của một âm tiết, nên chỉnh
luật chỉ cần sửa file dữ liệu, không phải sửa code test.

API cần có trong ``vietvoice.text.g2p``:
- ``Syllable``: dataclass với các trường onset, glide, nucleus, coda (str, rỗng nếu không có),
  tone (int 1..8, đã áp luật vùng), glottal_onset (bool).
- ``parse_syllable(word: str, dialect: str) -> Syllable``.
"""

import csv
from dataclasses import fields
from pathlib import Path

import pytest

from vietvoice.text.g2p import Syllable, parse_syllable

RULES_FILE = Path(__file__).parent / "data" / "g2p_rules.tsv"
FIELDS = {"onset", "glide", "nucleus", "coda", "tone", "glottal_onset"}


def load_rules() -> list[dict[str, str]]:
    lines = [ln for ln in RULES_FILE.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def expected_value(field: str, raw: str) -> object:
    if field == "tone":
        return int(raw)
    if field == "glottal_onset":
        return {"true": True, "false": False}[raw]
    return "" if raw == "-" else raw


RULES = load_rules()


def test_rules_file_is_well_formed():
    assert RULES, "g2p_rules.tsv không có dòng nào"
    for row in RULES:
        assert row["dialect"] in {"bac", "nam"}, row
        assert row["field"] in FIELDS, row
        expected_value(row["field"], row["expected"])


def test_syllable_has_the_fields_the_rules_use():
    assert {f.name for f in fields(Syllable)} == FIELDS


@pytest.mark.parametrize(
    "row",
    RULES,
    ids=[f"{r['dialect']}-{r['word']}-{r['field']}" for r in RULES],
)
def test_dialect_rule(row):
    syllable = parse_syllable(row["word"], row["dialect"])
    actual = getattr(syllable, row["field"])
    assert actual == expected_value(row["field"], row["expected"]), row["rule"]
