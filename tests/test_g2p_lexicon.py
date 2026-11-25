"""Từ điển ngoại lệ lexicon/{vùng}.tsv (đặc tả mục 3): tra trước luật, cách đọc ghi bằng âm tiết Việt.

API cần có trong ``parrot_tts.text.g2p``:
- ``LEXICON_DIR``: thư mục chứa ``bac.tsv``, ``nam.tsv``; test trỏ sang thư mục tạm.
- ``load_lexicon(dialect, directory=None) -> dict[str, tuple[str, ...]]``: từ (chữ thường, NFC) -> các âm tiết.
- ``INVENTORY_VERSION``: số phiên bản bộ âm vị.
"""

import unicodedata
from pathlib import Path

import pytest

from parrot_tts.text import g2p as g2p_module
from parrot_tts.text.g2p import DIALECTS, INVENTORY_VERSION, LEXICON_DIR, g2p, load_lexicon

HEADER = "# từ<TAB>cách đọc<TAB>ghi chú\n"


@pytest.fixture
def lexicon(tmp_path, monkeypatch):
    """Ghi file từ điển tạm cho mọi vùng và trỏ g2p sang đó."""
    monkeypatch.setattr(g2p_module, "LEXICON_DIR", tmp_path)

    def write(body: str, dialects: tuple[str, ...] = DIALECTS) -> Path:
        for dialect in dialects:
            (tmp_path / f"{dialect}.tsv").write_text(HEADER + body, encoding="utf-8")
        return tmp_path

    return write


# --- File từ điển thật trong repo ---


@pytest.mark.parametrize("dialect", DIALECTS)
def test_seeded_lexicon_loads_and_every_respelling_parses(dialect):
    entries = load_lexicon(dialect)
    assert len(entries) >= 25
    for word, syllables in entries.items():
        assert word == unicodedata.normalize("NFC", word).lower()
        assert syllables, word


@pytest.mark.parametrize("dialect", DIALECTS)
def test_seeded_rows_are_marked_draft(dialect):
    rows = [
        line.split("\t")
        for line in (LEXICON_DIR / f"{dialect}.tsv").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rows
    for row in rows:
        assert len(row) == 3 and "NHÁP" in row[2], row


@pytest.mark.parametrize("dialect", DIALECTS)
def test_spec_example_tiktok(dialect):
    assert g2p("TikTok", dialect) == g2p("tích tốc", dialect)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_multi_syllable_respelling_counts_as_syllables(dialect):
    tokens = g2p("xem tiktok nhé", dialect)
    assert tokens.count("_") == 3
    assert sum(t.startswith("T") and t[1:].isdigit() for t in tokens) == 4


@pytest.mark.parametrize("dialect", DIALECTS)
def test_punctuation_around_lexicon_word(dialect):
    assert g2p("tiktok, ok.", dialect) == g2p("tích tốc, ô kê.", dialect)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_unknown_word_still_raises(dialect):
    with pytest.raises(ValueError, match="tiktoks"):
        g2p("tiktoks", dialect)


def test_inventory_version():
    assert isinstance(INVENTORY_VERSION, int) and INVENTORY_VERSION == 1


# --- Hành vi tra từ điển (file tạm) ---


@pytest.mark.parametrize("dialect", DIALECTS)
def test_lexicon_beats_rules(lexicon, dialect):
    lexicon("ba\tbốn\tthử\n")
    assert g2p("ba", dialect) == g2p("bốn", dialect)
    assert g2p("ba", dialect) != ["ɓ", "a", "T1"]


@pytest.mark.parametrize("text", ["tiktok", "TIKTOK", "TikTok", "tIkToK"])
def test_lookup_is_case_insensitive(lexicon, text):
    lexicon("TikTok\ttích tốc\tthử\n")
    assert g2p(text, "bac") == g2p("tích tốc", "bac")


def test_lookup_matches_nfd_key_and_nfc_text(lexicon):
    lexicon(unicodedata.normalize("NFD", "Đà\tbốn\tthử\n"))
    assert g2p(unicodedata.normalize("NFC", "đà"), "bac") == g2p("bốn", "bac")
    assert g2p(unicodedata.normalize("NFD", "ĐÀ"), "bac") == g2p("bốn", "bac")


def test_respelling_is_read_with_each_dialect_rules(lexicon):
    lexicon("review\tri viu\tthử\nshare\tse\tthử\n")
    for word, respelling in [("review", "ri viu"), ("share", "se")]:
        assert g2p(word, "bac") == g2p(respelling, "bac")
        assert g2p(word, "nam") == g2p(respelling, "nam")
        assert g2p(word, "bac") != g2p(word, "nam")
    assert g2p("review", "bac")[0] == "z"
    assert g2p("share", "nam")[0] == "ʂ"


def test_note_column_is_optional_and_comments_blank_lines_skipped(lexicon):
    lexicon("\n# chú thích\nok\tô kê\n\n")
    assert load_lexicon("bac") == {"ok": ("ô", "kê")}


def test_load_lexicon_takes_directory_argument(tmp_path):
    (tmp_path / "nam.tsv").write_text(HEADER + "Fan\tphan\tthử\n", encoding="utf-8")
    assert load_lexicon("nam", tmp_path) == {"fan": ("phan",)}


def test_unknown_dialect_is_rejected(lexicon):
    lexicon("ok\tô kê\n")
    with pytest.raises(ValueError, match="hanoi"):
        load_lexicon("hanoi")


# --- File từ điển hỏng: báo lỗi nêu tên file và số dòng ---


@pytest.mark.parametrize(
    "bad_line",
    [
        "tiktok tích tốc",  # dùng dấu cách thay vì TAB
        "tiktok\ttích tốc\tghi chú\tthừa cột",
        "\ttích tốc\tthiếu từ",
        "tiktok\t \tthiếu cách đọc",
        "you tube\tdiu túp\ttừ có dấu cách",
        "tiktok\ttích tốk\tcách đọc không phải âm tiết Việt",
        "tiktok\ttích, tốc\tcách đọc có dấu câu",
        "4g\tbốn gờ\ttừ có chữ số",
    ],
)
def test_malformed_line_names_file_and_line(lexicon, bad_line):
    lexicon(f"ok\tô kê\n{bad_line}\n", dialects=("bac",))
    with pytest.raises(ValueError, match=r"bac\.tsv dòng 3"):
        load_lexicon("bac")


def test_bad_lexicon_breaks_g2p_loudly(lexicon):
    lexicon("tiktok\ttích tốk\n", dialects=("bac",))
    with pytest.raises(ValueError, match=r"bac\.tsv dòng 2"):
        g2p("ăn cơm", "bac")


def test_duplicate_word_is_an_error(lexicon):
    lexicon("tiktok\ttích tốc\nok\tô kê\nTikTok\ttíc tốc\n", dialects=("bac",))
    with pytest.raises(ValueError, match=r"bac\.tsv dòng 4.*tiktok.*dòng 2"):
        load_lexicon("bac")


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="nam.tsv"):
        load_lexicon("nam", tmp_path)
