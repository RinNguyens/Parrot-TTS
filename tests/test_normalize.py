"""Chuẩn hoá văn bản theo vùng (P1), đọc ca kiểm tra từ tests/data/normalize_cases.tsv.

Mỗi dòng TSV là một câu vào và đầu ra mong đợi, nên chỉnh cách đọc chỉ cần sửa file dữ liệu
(``tests/data/normalize_cases.tsv`` cho ca kiểm tra, ``normalize_data/`` cho luật).

API cần có trong ``vietvoice.text.normalize``:
- ``normalize(text, dialect, *, resolver=None, data=None) -> str``: chữ thường NFC, cách nhau
  một dấu cách, chỉ giữ dấu câu ``, . ? !``. Vùng lạ -> ``ValueError``.
- ``load_data(data_dir=DATA_DIR) -> NormalizeData``: đọc và kiểm tra file dữ liệu, lỗi -> ``ValueError``.
- ``AmbiguityResolver`` (Protocol) và ``AmbiguousSpan``: chỗ để LLM chọn cách đọc ca mơ hồ,
  tắt mặc định bằng ``ambiguity.use_resolver`` trong config.yaml.
"""

import csv
import re
import shutil
import time
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vietvoice.text.g2p import g2p, load_lexicon, split_text, word_tokens
from vietvoice.text.normalize import (
    DATA_DIR,
    AmbiguityResolver,
    AmbiguousSpan,
    load_data,
    normalize,
)

CASES_FILE = Path(__file__).parent / "data" / "normalize_cases.tsv"
DIALECTS = ("bac", "nam")
COLUMNS = ["dialect", "input", "expected", "category", "rule"]
CATEGORIES = {
    "so_dem",
    "so_thu_tu",
    "thap_phan",
    "phan_tram",
    "dien_thoai",
    "ngay",
    "gio",
    "tien_te",
    "don_vi",
    "viet_tat",
    "chu_hoa",
    "so_la_ma",
    "mo_ho",
    "url_email",
    "ngoai_lai",
    "ky_hieu",
    "dau_cau",
    "unicode",
    "hon_hop",
}
# Loại có cách đọc khác nhau giữa hai vùng: phải có dòng riêng cho cả bac lẫn nam.
REGIONAL_CATEGORIES = {"so_dem", "thap_phan", "ngay", "tien_te", "don_vi", "mo_ho", "hon_hop"}
# Loại mà đầu ra còn từ Latin (tên miền, từ mượn): G2P đọc qua từ điển ngoại lệ, không kiểm ở đây.
LOANWORD_CATEGORIES = {"url_email", "ngoai_lai"}
MIN_CASES = 150

# Chữ được phép ở đầu ra: a–z (từ mượn giữ nguyên), đ, và 12 nguyên âm tiếng Việt với 6 thanh.
_TONE_MARKS = ("", "\u0300", "\u0301", "\u0303", "\u0309", "\u0323")
_VOWELS = "aăâeêioôơuưy"
LETTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzđ" + "".join(unicodedata.normalize("NFC", v + t) for v in _VOWELS for t in _TONE_MARKS)
)
ALLOWED_OUTPUT = re.compile("^[" + re.escape("".join(sorted(LETTERS))) + " ,.?!]*$")


def load_cases() -> list[dict[str, str]]:
    lines = [ln for ln in CASES_FILE.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))
    for row in rows:
        row["input"] = row["input"].replace("\\n", "\n")
        row["expected"] = "" if row["expected"] == "-" else row["expected"]
    return rows


def expand(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    """Dòng ``*`` áp cho cả hai vùng."""
    return [(d, row) for row in rows for d in (DIALECTS if row["dialect"] == "*" else (row["dialect"],))]


ROWS = load_cases()
CASES = expand(ROWS)


def case_id(item: tuple[str, dict[str, str]]) -> str:
    dialect, row = item
    return f"{dialect}-{row['category']}-{row['input'][:30]}"


# --- File ca kiểm tra -------------------------------------------------------------------------


def test_cases_file_is_well_formed():
    assert len(ROWS) >= MIN_CASES, f"cần ít nhất {MIN_CASES} ca, mới có {len(ROWS)}"
    header = next(ln for ln in CASES_FILE.read_text(encoding="utf-8").splitlines() if not ln.startswith("#"))
    assert header.split("\t") == COLUMNS
    for row in ROWS:
        assert None not in row and all(row[c] is not None for c in COLUMNS), row
        assert row["dialect"] in {"bac", "nam", "*"}, row
        assert row["category"] in CATEGORIES, row
        assert row["input"].strip(), row
        assert row["rule"].strip(), row


def test_every_category_is_covered():
    assert {r["category"] for r in ROWS} == CATEGORIES


@pytest.mark.parametrize("category", sorted(REGIONAL_CATEGORIES))
def test_regional_categories_have_both_dialects(category):
    dialects = {r["dialect"] for r in ROWS if r["category"] == category}
    assert {"bac", "nam"} <= dialects, f"{category} thiếu dòng riêng cho một vùng: {dialects}"


def test_no_duplicate_cases():
    keys = [(d, r["input"]) for d, r in CASES]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"ca trùng: {sorted(dupes)}"


def test_expected_outputs_are_in_the_allowed_set():
    for dialect, row in CASES:
        assert ALLOWED_OUTPUT.match(row["expected"]), (dialect, row)
        assert row["expected"] == unicodedata.normalize("NFC", row["expected"]), row


# --- Từng ca ----------------------------------------------------------------------------------


@pytest.mark.parametrize("item", CASES, ids=[case_id(c) for c in CASES])
def test_case(item):
    dialect, row = item
    assert normalize(row["input"], dialect) == row["expected"], row["rule"]


@pytest.mark.parametrize("item", CASES, ids=[case_id(c) for c in CASES])
def test_idempotent(item):
    dialect, row = item
    once = normalize(row["input"], dialect)
    assert normalize(once, dialect) == once


@pytest.mark.parametrize(
    "item",
    [c for c in CASES if c[1]["category"] not in LOANWORD_CATEGORIES],
    ids=[case_id(c) for c in CASES if c[1]["category"] not in LOANWORD_CATEGORIES],
)
def test_output_is_readable_by_g2p(item):
    dialect, row = item
    g2p(normalize(row["input"], dialect), dialect)


# --- API --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["trung", "", "Bac", "north"])
def test_unknown_dialect_raises(dialect):
    with pytest.raises(ValueError, match="vùng"):
        normalize("xin chào", dialect)


def test_empty_text():
    assert normalize("", "bac") == ""
    assert normalize("   \n\t ", "nam") == ""


def test_output_is_nfc():
    decomposed = unicodedata.normalize("NFD", "Hoà bình, Đà Nẵng")
    out = normalize(decomposed, "bac")
    assert out == unicodedata.normalize("NFC", out) == "hoà bình, đà nẵng"


def test_regional_number_words_come_from_data():
    """Bắc/Nam khác nhau ở linh/lẻ và nghìn/ngàn; phần còn lại dùng chung."""
    assert normalize("1.105", "bac") == "một nghìn một trăm linh năm"
    assert normalize("1.105", "nam") == "một ngàn một trăm lẻ năm"


# --- Ca mơ hồ và AmbiguityResolver ------------------------------------------------------------


class FixedResolver:
    """Resolver giả: luôn chọn một cách đọc, ghi lại các ca được hỏi."""

    def __init__(self, choice: str | None):
        self.choice = choice
        self.spans: list[AmbiguousSpan] = []

    def resolve(self, span: AmbiguousSpan) -> str | None:
        self.spans.append(span)
        return self.choice


def resolver_data():
    return replace(load_data(), use_resolver=True)


def test_resolver_is_off_by_default():
    assert load_data().use_resolver is False


def test_fixed_resolver_satisfies_protocol():
    assert isinstance(FixedResolver("date"), AmbiguityResolver)


def test_resolver_rejected_when_config_turns_it_off():
    with pytest.raises(ValueError, match="use_resolver"):
        normalize("1/2", "bac", resolver=FixedResolver("fraction"))


def test_resolver_choice_overrides_rule():
    resolver = FixedResolver("fraction")
    assert normalize("Có 1/2 thôi", "bac", resolver=resolver, data=resolver_data()) == "có một phần hai thôi"
    (span,) = resolver.spans
    assert span.text[span.start : span.end] == "1/2"
    assert span.rule_choice == "date"
    assert set(span.options) == {"date", "fraction", "ratio"}


def test_resolver_returning_none_keeps_rule():
    resolver = FixedResolver(None)
    assert normalize("1/2", "nam", resolver=resolver, data=resolver_data()) == "ngày một tháng hai"
    assert len(resolver.spans) == 1


def test_resolver_for_dash_and_colon():
    data = resolver_data()
    assert normalize("3-4", "bac", resolver=FixedResolver("score"), data=data) == "ba bốn"
    assert normalize("14:30", "bac", resolver=FixedResolver("score"), data=data) == "mười bốn ba mươi"


def test_resolver_invalid_choice_raises():
    with pytest.raises(ValueError, match="time"):
        normalize("1/2", "bac", resolver=FixedResolver("time"), data=resolver_data())


def test_resolver_not_asked_for_unambiguous_text():
    resolver = FixedResolver("fraction")
    normalize("Ngày 15/1/2027 lúc 14h30, giá 50.000đ.", "bac", resolver=resolver, data=resolver_data())
    assert resolver.spans == []


# --- File dữ liệu lỗi -------------------------------------------------------------------------


@pytest.fixture
def data_copy(tmp_path: Path) -> Path:
    target = tmp_path / "normalize_data"
    shutil.copytree(DATA_DIR, target)
    return target


def append_line(path: Path, line: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")


def test_default_data_loads():
    data = load_data()
    assert set(data.dialects) == set(DIALECTS)


def test_bad_reading_in_data_raises(data_copy: Path):
    append_line(data_copy / "abbreviations.tsv", "*\tXYZW\tChữ Hoa 1")
    with pytest.raises(ValueError, match="abbreviations.tsv"):
        load_data(data_copy)


def test_missing_dialect_reading_raises(data_copy: Path):
    append_line(data_copy / "units.tsv", "bac\tzz\tdét dét")
    with pytest.raises(ValueError, match="nam"):
        load_data(data_copy)


def test_duplicate_key_in_data_raises(data_copy: Path):
    append_line(data_copy / "letters.tsv", "*\tA\ta")
    with pytest.raises(ValueError, match="letters.tsv"):
        load_data(data_copy)


def test_bad_config_raises(data_copy: Path):
    config = data_copy / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace("tone_style: new", "tone_style: xyz"), "utf-8")
    with pytest.raises(ValueError, match="tone_style"):
        load_data(data_copy)


def test_old_tone_style_from_config():
    data = replace(load_data(), tone_style="old")
    assert normalize("hoà thuận, thuỷ, khoẻ, quý", "bac", data=data) == "hòa thuận, thủy, khỏe, quý"


# --- Chuỗi ngẫu nhiên và hiệu năng ------------------------------------------------------------

_CHARS = list("0123456789 .,/-:;%$€đkKmhgsVWB²³°@+()!?…\"'\n#&_~TPHCMUBNDĐXIVabcxyzàáảãạăâêôơưÀỦỶ")
_CHARS += ["😀", "❤", "\ufe0f", "\u0301", "\u0300", "\u00a0", "ñ", "Ð", "中", "ß", "İ", "٣", "\t", "\r"]
_TOKENS = ["ngày", "tháng", "thứ", "tỉ số", "điểm", "thế kỷ", "XXI", "lúc", "giá", "1/2", "3-4", "14:30", "10/10"]
_TOKENS += ["0912345678", "+84 912 345 678", "15/1/2027", "2.500", "2,5", "2.5", "50k", "$5", "a@b.vn", "UBND"]
_TOKENS += ["TP.HCM", "v.v.", "km/h", "https://x.vn/a-b", "1.000.000", "-5°C", "2tr", "thứ IV", "OK", "hòa"]

TEXT = st.one_of(
    st.text(max_size=40),
    st.text(alphabet=st.sampled_from(_CHARS), max_size=40),
    st.lists(st.sampled_from(_TOKENS + _CHARS), max_size=12).map(" ".join),
    st.lists(st.sampled_from(_TOKENS + _CHARS), max_size=12).map("".join),
)


@settings(max_examples=10_000, deadline=None)
@given(text=TEXT, dialect=st.sampled_from(DIALECTS))
def test_random_text_never_raises_and_output_is_clean(text: str, dialect: str):
    out = normalize(text, dialect)
    assert ALLOWED_OUTPUT.match(out), repr(out)
    assert out == out.strip() and "  " not in out, repr(out)
    assert out == unicodedata.normalize("NFC", out)
    assert_g2p_readable(out, dialect)


PLAIN_LATIN = re.compile("^[a-z]+$")


def assert_g2p_readable(out: str, dialect: str) -> None:
    """G2P đọc được đầu ra; chỉ được phép sót từ Latin thuần (a–z) chưa có trong từ điển ngoại lệ."""
    lexicon = load_lexicon(dialect)
    for piece in split_text(out):
        try:
            word_tokens(piece, dialect)
        except ValueError:
            assert PLAIN_LATIN.match(piece) and piece not in lexicon, f"G2P không đọc được {piece!r} trong {out!r}"


REALISTIC = (
    "Sáng 15/1/2027, UBND TP.HCM tổ chức hội nghị tại số 86 Lê Thánh Tôn; có 2.500 đại biểu tham dự. "
    "Giá vé từ 50k đến 1,5 triệu đồng, tăng 15% so với năm 2026. Liên hệ 0912 345 678 hoặc info@vietvoice.vn. "
    "Nhiệt độ 25-30°C, gió 20km/h. Đội nhà thắng tỉ số 3-1 lúc 21h15 😀 v.v. Video TikTok đạt 10/10 điểm! "
    "Thế kỷ XXI, lần thứ 4, phòng 105, diện tích 120m², 1/2 số người đồng ý. "
)


def test_performance_5000_chars_under_50ms():
    text = (REALISTIC * (5000 // len(REALISTIC) + 1))[:5000]
    for dialect in DIALECTS:
        normalize(text, dialect)  # nạp dữ liệu và biên dịch regex trước khi đo
        best = min(_timed(text, dialect) for _ in range(5))
        assert best < 0.05, f"{dialect}: {best * 1000:.1f} ms cho 5.000 ký tự"


def _timed(text: str, dialect: str) -> float:
    start = time.perf_counter()
    normalize(text, dialect)
    return time.perf_counter() - start
