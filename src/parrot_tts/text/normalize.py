"""Chuẩn hoá văn bản theo vùng (P1): số, ngày, giờ, tiền, đơn vị, viết tắt, URL -> chữ đọc được.

Đầu ra: chữ thường NFC, các từ cách nhau một dấu cách, chỉ giữ dấu câu ``, . ? !`` (dính vào từ
đứng trước). Từ Latin không phải viết tắt (TikTok, video) giữ nguyên, viết thường: G2P đọc chúng
qua từ điển ngoại lệ. Chữ không phải Latin/tiếng Việt, emoji và ký hiệu lạ bị bỏ.

Mọi từ đọc và luật chỉnh được nằm trong ``normalize_data/``, code không chứa từ đọc nào:
- ``numbers.yaml``: từ đọc số; phần khác vùng (linh/lẻ, nghìn/ngàn) ghi đè phần chung;
- ``abbreviations.tsv``, ``units.tsv``, ``letters.tsv``, ``symbols.tsv``, ``url.tsv``: bảng tra theo vùng;
- ``config.yaml``: ngưỡng, từ ngữ cảnh, đổi dấu câu, kiểu bỏ dấu, công tắc resolver.

Luật đọc số:
- Dấu chấm + đúng 3 chữ số là phân cách nghìn (2.500); dấu phẩy là thập phân (2,5). Từ hai dấu
  phẩy nghìn trở lên kiểu Anh (1,000,000) cũng là phân cách nghìn.
- "2.5": có đơn vị, tiền tệ, % theo sau, hoặc từ nghìn/triệu/tỷ đi sau, đọc "phẩy"; còn lại đọc
  "chấm" (số phiên bản, mục 2.5.1).
- Nhóm ba chữ số rỗng bị bỏ; nhóm đứng sau nhóm lớn đọc đủ "không trăm" (1.005 -> một nghìn không
  trăm linh năm; 1.000.005 -> một triệu không trăm linh năm).

Ca mơ hồ (xét theo thứ tự; danh sách từ ngữ cảnh ở ``config.yaml``, mục ``context``):
- ``a/b``: (1) từ trước thuộc date_before hoặc month -> tháng/năm (b có 4 chữ số) hoặc ngày/tháng;
  (2) ratio_before, ratio_after -> "a trên b"; (3) fraction_before, fraction_after -> "a phần b";
  (4) không có ngữ cảnh: tháng/năm, rồi ngày/tháng nếu hợp lệ, rồi phân số, cuối cùng "a trên b".
- ``a-b`` (không kèm đơn vị): từ trước thuộc score_before -> tỉ số "a b"; còn lại là khoảng "a đến b".
- ``a:b``: score_before -> tỉ số; giờ hợp lệ -> giờ phút; còn lại đọc như tỉ số.
Khi ``ambiguity.use_resolver`` bật, mỗi ca có hơn một cách đọc được hỏi ``AmbiguityResolver``
(ví dụ một LLM) trước; resolver trả ``None`` thì giữ lựa chọn của luật.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, get_args, runtime_checkable

import yaml

from parrot_tts.text.g2p import DIALECTS, load_lexicon, parse_syllable

DATA_DIR = Path(__file__).parent / "normalize_data"

AmbiguityKind = Literal["date", "month_year", "fraction", "ratio", "range", "score", "time"]
AMBIGUITY_KINDS: tuple[str, ...] = get_args(AmbiguityKind)

PUNCTUATION = ",.?!"
_TONE_MARKS = ("", "\u0300", "\u0301", "\u0303", "\u0309", "\u0323")
_VOWELS = "aăâeêioôơuưy"
# Chữ được phép ở đầu ra: a–z (từ mượn), đ, 12 nguyên âm tiếng Việt với 6 thanh.
LETTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzđ" + "".join(unicodedata.normalize("NFC", v + t) for v in _VOWELS for t in _TONE_MARKS)
)
_VIETNAMESE_CHARS = LETTERS | {c.upper() for c in LETTERS}
_LETTER_CLASS = "[" + re.escape("".join(sorted(LETTERS))) + "]"
_READING = re.compile(f"{_LETTER_CLASS}+(?: {_LETTER_CLASS}+)*")

_CONTEXT_KEYS = (
    "ordinal",
    "month",
    "roman",
    "decimal_scale_after",
    "date_before",
    "date_no_prefix",
    "ratio_before",
    "ratio_after",
    "fraction_before",
    "fraction_after",
    "score_before",
    "currency_before",
)
_TABLES = ("abbreviations", "units", "letters", "symbols", "url")

_WORD = r"[^\W\d_]"
_TAIL_WORDS = re.compile(rf"((?:{_WORD}+ +){{0,2}}{_WORD}+) *$")
_HEAD_WORD = re.compile(rf" *({_WORD}+)")
_URL_PIECE = re.compile(rf"{_WORD}+|[0-9]+|\S")
_PREPASS = re.compile(r"[^\x20-\x7e\n]")
_THOUSANDS = re.compile(r"[1-9][0-9]{0,2}(?:\.[0-9]{3})+(?:,[0-9]+)?")
_EN_THOUSANDS = re.compile(r"[0-9]{1,3}(?:,[0-9]{3}){2,}")
_ROMAN = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

# Kiểu bỏ dấu cho vần mở oa, oe, uy (dạng NFD): cũ hòa/thủy <-> mới hoà/thuỷ.
_TONE = "[\u0300\u0301\u0303\u0309\u0323]"
_NO_VOWEL_BEFORE = r"(?<![aeiouy\u0300-\u036f])"
_NO_VOWEL_OR_Q_BEFORE = r"(?<![aeiouyq\u0300-\u036f])"
_WORD_END = r"(?![^\W\d_]|[\u0300-\u036f])"
_RESTYLE = {
    "new": (
        (re.compile(f"{_NO_VOWEL_BEFORE}o({_TONE})([ae]){_WORD_END}"), r"o\2\1"),
        (re.compile(f"{_NO_VOWEL_OR_Q_BEFORE}u({_TONE})y{_WORD_END}"), r"uy\1"),
    ),
    "old": (
        (re.compile(f"{_NO_VOWEL_BEFORE}o([ae])({_TONE}){_WORD_END}"), r"o\2\1"),
        (re.compile(f"{_NO_VOWEL_OR_Q_BEFORE}uy({_TONE}){_WORD_END}"), r"u\1y"),
    ),
}

_NUM = (
    r"(?:[1-9][0-9]{0,2}(?:\.[0-9]{3})+(?:,[0-9]+)?(?![0-9]|\.[0-9])"
    r"|[0-9]{1,3}(?:,[0-9]{3}){2,}(?![0-9]|,[0-9])"
    r"|[0-9]+(?:(?:\.[0-9]+)+|,[0-9]+)?)"
)
_URL_END = r"""[^\s.,!?;:)\]}'"»”]"""


@dataclass(frozen=True)
class AmbiguousSpan:
    """Một chỗ có nhiều cách đọc, đưa cho ``AmbiguityResolver`` chọn."""

    text: str  # toàn bộ văn bản sau bước làm sạch ký tự (NFC), để resolver xem ngữ cảnh
    start: int
    end: int  # text[start:end] là chỗ mơ hồ, ví dụ "1/2"
    options: tuple[str, ...]  # các cách đọc hợp lệ, lấy trong AMBIGUITY_KINDS
    rule_choice: str  # cách luật đã chọn


@runtime_checkable
class AmbiguityResolver(Protocol):
    """Bộ chọn cách đọc ca mơ hồ (ví dụ một LLM). Trả ``None`` để giữ lựa chọn của luật."""

    def resolve(self, span: AmbiguousSpan) -> str | None: ...


@dataclass(frozen=True)
class NumberWords:
    """Từ đọc số của một vùng, lấy từ numbers.yaml."""

    digits: tuple[str, ...]
    ten: str
    tens: str
    hundred: str
    thousand: str
    million: str
    billion: str
    zero_tens: str
    after_ten: dict[int, str]
    after_tens: dict[int, str]
    ordinal: dict[int, str]
    month: dict[int, str]
    minus: str
    decimal_comma: str
    decimal_dot: str
    fraction: str
    ratio: str
    range: str
    day: str
    month_word: str
    year: str
    hour: str
    minute: str
    second: str


@dataclass(frozen=True)
class DialectData:
    """Mọi cách đọc của một vùng."""

    numbers: NumberWords
    abbreviations: dict[str, str]
    units: dict[str, str]
    letters: dict[str, str]
    symbols: dict[str, str]
    url: dict[str, str]


@dataclass(frozen=True)
class NormalizeData:
    """Dữ liệu và cấu hình đã đọc, kiểm tra từ ``normalize_data/``."""

    dialects: dict[str, DialectData]
    use_resolver: bool
    tone_style: str  # "new" | "old"
    max_digits: int
    max_leading_zero_digits: int
    max_fraction_digits: int
    max_spell_letters: int
    phone_groups: dict[int, tuple[int, ...]]
    phone_separator: str
    max_hour: int
    context: dict[str, tuple[tuple[str, ...], ...]]
    punctuation: dict[str, str]
    punctuation_strength: dict[str, int]  # số nhỏ hơn là dấu mạnh hơn
    char_map: dict[int, str]
    resolution_unit: str  # 4K: chữ đọc theo tên chữ cái thay vì đơn vị
    resolution_numbers: frozenset[int]
    compact_amount: dict[str, int]  # 1tr5, 2k5: hệ số của chữ ở giữa
    pattern: re.Pattern[str]


def normalize(
    text: str,
    dialect: str,
    *,
    resolver: AmbiguityResolver | None = None,
    data: NormalizeData | None = None,
) -> str:
    """Văn bản thô -> chữ đọc được theo ``dialect`` ("bac" hoặc "nam")."""
    data = data or _default_data()
    if dialect not in data.dialects:
        raise ValueError(f"vùng {dialect!r} không hợp lệ; chỉ có: {', '.join(data.dialects)}")
    if resolver is not None and not data.use_resolver:
        raise ValueError("resolver đang tắt; bật ambiguity.use_resolver trong normalize_data/config.yaml")
    clean = _PREPASS.sub(_prepass_char, unicodedata.normalize("NFC", text.translate(data.char_map)))
    tokens = _Run(clean, data, dialect, resolver).tokens()
    return _restyle(_assemble(tokens, data), data.tone_style)


def load_data(data_dir: Path = DATA_DIR) -> NormalizeData:
    """Đọc và kiểm tra mọi file trong ``data_dir``. File sai -> ``ValueError`` nêu rõ file và dòng."""
    config = _load_yaml(data_dir / "config.yaml")
    numbers = _load_yaml(data_dir / "numbers.yaml")
    tables = {name: _load_table(data_dir / f"{name}.tsv") for name in _TABLES}
    try:
        return _build(config, numbers, tables)
    except (KeyError, TypeError) as err:
        raise ValueError(f"normalize_data sai cấu trúc: {err!r}") from err


@cache
def _default_data() -> NormalizeData:
    return load_data(DATA_DIR)


# --- Một lần chuẩn hoá ------------------------------------------------------------------------


class _Run:
    """Quét văn bản một lượt bằng regex tổng, mỗi loại khớp có một hàm đọc."""

    def __init__(self, text: str, data: NormalizeData, dialect: str, resolver: AmbiguityResolver | None):
        self.text = text
        self.data = data
        self.dd = data.dialects[dialect]
        self.nw = self.dd.numbers
        self.lexicon = load_lexicon(dialect)  # từ có ở đây để nguyên cho G2P (từ điển thắng luật)
        self.resolver = resolver
        self.handlers: dict[str, Callable[[re.Match[str]], list[str]]] = {
            "url": self._url,
            "email": self._url,
            "domain": self._url,
            "dmy": self._date,
            "ymd": self._date,
            "phone": self._phone,
            "timeh": self._time_h,
            "colon": self._colon,
            "slash": self._slash,
            "compact": self._compact,
            "quant": self._quant,
            "abbr": self._abbr,
            "hyph": self._hyph,
            "word": self._word,
            "newline": self._newline,
            "other": self._other,
        }

    def tokens(self) -> list[str]:
        out: list[str] = []
        for m in self.data.pattern.finditer(self.text):
            out += self.handlers[m.lastgroup or "other"](m)
        return out

    # Ngữ cảnh: từ ngay trước / ngay sau chỗ khớp, chỉ cách bằng dấu cách.

    def _before(self, start: int, key: str) -> bool:
        found = _TAIL_WORDS.search(self.text, max(0, start - 40), start)
        if not found:
            return False
        words = tuple(_context_form(found.group(1)).split())
        return any(words[-len(p) :] == p for p in self.data.context[key])

    def _after(self, end: int, key: str) -> bool:
        found = _HEAD_WORD.match(self.text, end, end + 40)
        return bool(found) and (_context_form(found.group(1)),) in self.data.context[key]

    def _choose(self, m: re.Match[str], options: tuple[str, ...], rule: str) -> str:
        if self.resolver is None or len(options) < 2:
            return rule
        span = AmbiguousSpan(text=self.text, start=m.start(), end=m.end(), options=options, rule_choice=rule)
        choice = self.resolver.resolve(span)
        if choice is None:
            return rule
        if choice not in options:
            raise ValueError(f"resolver chọn {choice!r} cho {m.group()!r}; chỉ được chọn: {', '.join(options)}")
        return choice

    # Đọc số.

    def _int_words(self, n: int, full: bool = False) -> list[str]:
        if n == 0 and not full:
            return [self.nw.digits[0]]
        words: list[str] = []
        if n >= 10**9:
            high, n = divmod(n, 10**9)
            words += self._int_words(high, full) + [self.nw.billion]
            if n == 0:
                return words
            full = True
        millions, rest = divmod(n, 10**6)
        thousands, units = divmod(rest, 1000)
        for value, scale in ((millions, self.nw.million), (thousands, self.nw.thousand), (units, "")):
            if value:
                words += self._triple(value, full)
                if scale:
                    words.append(scale)
                full = True
        return words

    def _triple(self, value: int, full: bool) -> list[str]:
        """Một nhóm 0..999; ``full`` là nhóm đứng sau nhóm lớn hơn nên đọc đủ hàng trăm."""
        nw = self.nw
        hundreds, rest = divmod(value, 100)
        tens, units = divmod(rest, 10)
        words: list[str] = []
        if hundreds or full:
            words += [nw.digits[hundreds], nw.hundred]
        if tens == 0:
            if units:
                words += [nw.zero_tens, nw.digits[units]] if words else [nw.digits[units]]
        elif tens == 1:
            words += [nw.ten] + ([nw.after_ten.get(units, nw.digits[units])] if units else [])
        else:
            words += [nw.digits[tens], nw.tens] + ([nw.after_tens.get(units, nw.digits[units])] if units else [])
        return words

    def _digits(self, digits: str) -> list[str]:
        return [self.nw.digits[int(d)] for d in digits]

    def _int_str(self, digits: str) -> list[str]:
        """Chuỗi chữ số nguyên: quá dài hoặc có số 0 đầu dài thì đọc từng chữ số."""
        too_long = len(digits) > self.data.max_digits
        leading_zero = len(digits) > max(1, self.data.max_leading_zero_digits) and digits[0] == "0"
        return self._digits(digits) if too_long or leading_zero else self._int_words(int(digits))

    def _fraction_digits(self, digits: str) -> list[str]:
        """Phần sau dấu thập phân: số 0 đầu đọc "không", phần còn lại đọc như số nếu đủ ngắn."""
        if len(digits) > self.data.max_fraction_digits:
            return self._digits(digits)
        rest = digits.lstrip("0")
        return self._digits(digits[: len(digits) - len(rest)]) + (self._int_words(int(rest)) if rest else [])

    def _number(self, s: str, quantity: bool) -> list[str]:
        """Một số có thể có phân cách nghìn, thập phân hoặc nhiều dấu chấm (phiên bản)."""
        nw = self.nw
        if _THOUSANDS.fullmatch(s):
            integer, _, frac = s.partition(",")
            words = self._int_str(integer.replace(".", ""))
            return words + ([nw.decimal_comma] + self._fraction_digits(frac) if frac else [])
        if _EN_THOUSANDS.fullmatch(s):
            return self._int_str(s.replace(",", ""))
        if "," in s:
            integer, frac = s.split(",", 1)
            return self._int_str(integer) + [nw.decimal_comma] + self._fraction_digits(frac)
        if "." in s:
            parts = s.split(".")
            if len(parts) == 2 and quantity:
                return self._int_str(parts[0]) + [nw.decimal_comma] + self._fraction_digits(parts[1])
            words = self._int_str(parts[0])
            for part in parts[1:]:
                words += [nw.decimal_dot] + self._fraction_digits(part)
            return words
        return self._int_str(s)

    def _month(self, n: int) -> list[str]:
        return [self.nw.month[n]] if n in self.nw.month else self._int_words(n)

    # Các loại khớp.

    def _url(self, m: re.Match[str]) -> list[str]:
        words: list[str] = []
        for piece in _URL_PIECE.findall(re.sub(r"^https?://", "", m.group())):
            if piece[0] in "0123456789":
                words += self._digits(piece)
            elif piece.isalpha():
                lower = piece.lower()
                words += [self.dd.url[lower]] if lower in self.dd.url else self._read_word(piece, m.start())
            elif piece in self.dd.url:
                words.append(self.dd.url[piece])
        return words

    def _date(self, m: re.Match[str]) -> list[str]:
        if m.lastgroup == "dmy":
            day, month, year = m.group("dmy_d"), m.group("dmy_m"), m.group("dmy_y")
        else:
            year, month, day = m.group("ymd_y"), m.group("ymd_m"), m.group("ymd_d")
        d, mo = int(day), int(month)
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return [w for part in re.findall("[0-9]+", m.group()) for w in self._int_str(part)]
        nw = self.nw
        prefix = [] if self._before(m.start(), "date_no_prefix") else [nw.day]
        return prefix + self._int_words(d) + [nw.month_word] + self._month(mo) + [nw.year] + self._int_str(year)

    def _phone(self, m: re.Match[str]) -> list[str]:
        span = m.group()
        groups = re.findall("[0-9]+", span)
        prefix: list[str] = []
        if span.startswith("+"):
            groups = groups[1:] if groups[0] == "84" else [groups[0][2:], *groups[1:]]
            prefix = [self.dd.symbols["+"], *self._digits("84"), self.data.phone_separator]
        total = sum(map(len, groups))
        hotline = span[0] == "1"
        if not (hotline or (prefix and total in (9, 10)) or (not prefix and total in (10, 11))):
            return [w for g in groups for w in self._int_str(g)]
        if len(groups) == 1:
            run = groups[0]
            sizes = (4, len(run) - 4) if hotline else self.data.phone_groups.get(len(run), (len(run),))
            groups, pos = [], 0
            for size in sizes:
                groups.append(run[pos : pos + size])
                pos += size
        words = prefix
        for i, group in enumerate(groups):
            words += ([self.data.phone_separator] if i else []) + self._digits(group)
        return words

    def _time_h(self, m: re.Match[str]) -> list[str]:
        words = self._hour_words(m, "th")
        if m.group("th2_h") is not None:
            words += [self.nw.range] + self._hour_words(m, "th2")
        return words

    def _hour_words(self, m: re.Match[str], prefix: str) -> list[str]:
        nw = self.nw
        minute = m.group(f"{prefix}_m") or m.group(f"{prefix}_g")
        words = self._int_words(int(m.group(f"{prefix}_h"))) + [nw.hour]
        if minute and int(minute):
            words += self._int_words(int(minute)) + [nw.minute]
        return words

    def _colon(self, m: re.Match[str]) -> list[str]:
        words = self._colon_side(m, "c")
        if m.group("c2_a") is not None:
            words += [self.nw.range] + self._colon_side(m, "c2")
        return words

    def _colon_side(self, m: re.Match[str], prefix: str) -> list[str]:
        a, b, c = m.group(f"{prefix}_a"), m.group(f"{prefix}_b"), m.group(f"{prefix}_c")
        valid = (
            len(b) == 2
            and int(a) <= self.data.max_hour
            and int(b) <= 59
            and (c is None or (len(c) == 2 and int(c) <= 59))
        )
        options = ("time", "score") if valid else ("score",)
        rule = "score" if self._before(m.start(), "score_before") else options[0]
        if self._choose(m, options, rule) == "score":
            return [w for part in (a, b, c) if part is not None for w in self._int_str(part)]
        nw = self.nw
        minute, second = int(b), int(c or "0")
        words = self._int_words(int(a)) + [nw.hour]
        if minute or second:
            words += self._int_words(minute) + [nw.minute]
        if second:
            words += self._int_words(second) + [nw.second]
        return words

    def _slash(self, m: re.Match[str]) -> list[str]:
        a, b = m.group("s_a"), m.group("s_b")
        ia, ib = int(a), int(b)
        options: list[str] = []
        if len(a) <= 2 and len(b) == 4 and 1 <= ia <= 12:
            options.append("month_year")
        if len(a) <= 2 and len(b) <= 2 and 1 <= ia <= 31 and 1 <= ib <= 12:
            options.append("date")
        if ib:
            options.append("fraction")
        options.append("ratio")
        start, end = m.start(), m.end()
        if options[0] in ("month_year", "date") and (
            self._before(start, "date_before") or self._before(start, "month")
        ):
            rule = options[0]
        elif self._before(start, "ratio_before") or self._after(end, "ratio_after"):
            rule = "ratio"
        elif "fraction" in options and (self._before(start, "fraction_before") or self._after(end, "fraction_after")):
            rule = "fraction"
        else:
            rule = options[0]
        kind = self._choose(m, tuple(options), rule)
        nw = self.nw
        if kind == "month_year":
            prefix = [] if self._before(start, "month") else [nw.month_word]
            return prefix + self._month(ia) + [nw.year] + self._int_str(b)
        if kind == "date":
            prefix = [] if self._before(start, "date_no_prefix") else [nw.day]
            return prefix + self._int_words(ia) + [nw.month_word] + self._month(ib)
        middle = nw.fraction if kind == "fraction" else nw.ratio
        return self._int_str(a) + [middle] + self._int_str(b)

    def _compact(self, m: re.Match[str]) -> list[str]:
        """1tr5 -> 1.500.000, 2k5 -> 2.500: chữ số sau chữ là các chữ số đầu của phần lẻ."""
        scale = self.data.compact_amount[m.group("cm_u")]
        rest = int(m.group("cm_b").ljust(3, "0")) * (scale // 1000)
        return self._int_words(int(m.group("cm_a")) * scale + rest)

    def _quant(self, m: re.Match[str]) -> list[str]:
        pre, pre_b, neg, a, b = (m.group(g) for g in ("q_pre", "q_pre_b", "q_neg", "q_a", "q_b"))
        unit_a, unit_b = m.group("q_ua") or m.group("q_uas"), m.group("q_ub") or m.group("q_ubs")
        start, nw, units = m.start(), self.nw, self.dd.units
        quantity = bool(pre or pre_b or unit_a or unit_b) or self._after(m.end(), "decimal_scale_after")
        words = [nw.minus] if neg else []
        if b is not None:
            plain = not (pre or pre_b or unit_a or unit_b or neg)
            options = ("range", "score") if plain else ("range",)
            rule = "score" if plain and self._before(start, "score_before") else "range"
            kind = self._choose(m, options, rule)
            words += self._number(a, quantity) + ([units[unit_a]] if unit_a else [])
            words += [units[pre]] if pre and pre_b else []
            words += [nw.range] if kind == "range" else []
            words += self._number(b, quantity) + ([units[unit_b]] if unit_b else [])
            return words + ([units[pre_b or pre]] if pre_b or pre else [])
        plain = not (pre or unit_a or neg)
        if (
            not (pre or neg)
            and m.group("q_ua") == self.data.resolution_unit
            and a.isdigit()
            and len(a) <= 2
            and int(a) in self.data.resolution_numbers
            and not self._before(start, "currency_before")
        ):
            return self._int_words(int(a)) + [self.dd.letters[self.data.resolution_unit]]
        if plain and a.isdigit() and len(a) <= self.data.max_digits and self._before(start, "ordinal"):
            n = int(a)
            words += [nw.ordinal[n]] if n in nw.ordinal else self._int_str(a)
        elif plain and a.isdigit() and len(a) <= 2 and 1 <= int(a) <= 12 and self._before(start, "month"):
            words += self._month(int(a))
        else:
            words += self._number(a, quantity)
        return words + ([units[unit_a]] if unit_a else []) + ([units[pre]] if pre else [])

    def _abbr(self, m: re.Match[str]) -> list[str]:
        return [self.dd.abbreviations[m.group()]]

    def _word(self, m: re.Match[str]) -> list[str]:
        token = m.group()
        if token.isalpha():
            return self._read_word(token, m.start())
        # \w còn khớp ký tự số không phải chữ số (², ½, Ⅻ): bỏ chúng, đọc phần chữ còn lại.
        return [
            w
            for piece in "".join(c if c.isalpha() else " " for c in token).split()
            for w in self._read_word(piece, m.start())
        ]

    def _hyph(self, m: re.Match[str]) -> list[str]:
        """Wi-Fi, K-pop, e-mail: ghép thành một từ Latin để G2P tra từ điển ngoại lệ.

        Không ghép khi mọi phần viết hoa (USB-C: đánh vần từng phần) hoặc mọi phần là âm tiết
        tiếng Việt (Ba-Na), trừ khi từ ghép có trong từ điển ngoại lệ.
        """
        parts = m.group().split("-")
        joined = "".join(parts).lower()
        all_caps = all(p.isupper() for p in parts)
        all_syllables = all(_is_syllable(p.lower()) for p in parts)
        if joined in self.lexicon or not (all_caps or all_syllables):
            return self._read_word(joined, m.start())
        return [w for part in parts for w in self._read_word(part, m.start())]

    def _read_word(self, token: str, start: int) -> list[str]:
        """Một từ: số La Mã có ngữ cảnh, từ điển ngoại lệ, âm tiết Việt, đơn vị đứng riêng, đánh vần, từ Latin."""
        if token.isupper() and _ROMAN.fullmatch(token) and self._before(start, "roman"):
            value = _roman_value(token)
            if self._before(start, "ordinal") and value in self.nw.ordinal:
                return [self.nw.ordinal[value]]
            return self._int_words(value)
        lower = token.lower()
        if lower in self.lexicon or _is_syllable(lower):
            return [lower]
        if len(lower) > 1 and lower in self.dd.units and not _has_vowel(lower):
            return [self.dd.units[lower]]  # "còn mấy km nữa"
        if self._should_spell(token, lower):
            return [self.dd.letters[c.upper()] for c in token]
        # Từ Latin để G2P tra từ điển; chữ có dấu nhưng không phải âm tiết (café) bỏ dấu cho thành a–z.
        return [_ascii_fold(lower)]

    def _should_spell(self, token: str, lower: str) -> bool:
        """Đánh vần chữ cái lẻ, chữ không có nguyên âm (vtv, mp) và chữ hoa ngắn (ABC); đã biết không phải âm tiết."""
        if not all(c.upper() in self.dd.letters for c in token):
            return False
        if len(token) == 1 or not _has_vowel(lower):
            return True
        return token.isupper() and len(token) <= self.data.max_spell_letters

    def _newline(self, m: re.Match[str]) -> list[str]:
        return [self.data.punctuation["\n"]] if "\n" in self.data.punctuation else []

    def _other(self, m: re.Match[str]) -> list[str]:
        char = m.group()
        if char in PUNCTUATION:
            return [char]
        if char == "-":
            text, start, end = self.text, m.start(), m.end()
            spaced = start > 0 and text[start - 1] == " " and end < len(text) and text[end] == " "
            return [self.data.punctuation[" - "]] if spaced and " - " in self.data.punctuation else []
        if char in self.data.punctuation:
            return [self.data.punctuation[char]]
        if char in self.dd.symbols:
            return [self.dd.symbols[char]]
        return []


def _roman_value(token: str) -> int:
    values = [_ROMAN_VALUES[c] for c in token]
    return sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values))


@lru_cache(maxsize=8192)
def _is_syllable(word: str) -> bool:
    """Có phải một âm tiết tiếng Việt viết đúng chính tả (không phụ thuộc vùng)."""
    try:
        parse_syllable(word, DIALECTS[0])
    except ValueError:
        return False
    return True


def _ascii_fold(word: str) -> str:
    """Bỏ mọi dấu: café -> cafe, đắk -> dak."""
    return "".join(c for c in unicodedata.normalize("NFD", word.replace("đ", "d")) if unicodedata.category(c)[0] != "M")


def _has_vowel(word: str) -> bool:
    return any(c in "aeiouy" for c in unicodedata.normalize("NFD", word))


@lru_cache(maxsize=65536)
def _fold_char(char: str) -> str:
    """Một ký tự ngoài ASCII in được: giữ chữ Việt, bỏ dấu lạ, đổi chữ số và khoảng trắng lạ."""
    if char in _VIETNAMESE_CHARS:
        return char
    category = unicodedata.category(char)
    if category[0] == "M":
        return ""
    if category == "Nd":
        return str(unicodedata.decimal(char))
    if char in "\r\x85  ":
        return "\n"
    if char.isspace():
        return " "
    if char.isalpha():
        base = unicodedata.normalize("NFD", char)[0]
        return base if base.isascii() and base.isalpha() else " "
    return char


def _prepass_char(m: re.Match[str]) -> str:
    return _fold_char(m.group())


def _assemble(tokens: list[str], data: NormalizeData) -> str:
    """Ghép token: từ cách nhau một dấu cách, dấu câu dính vào từ trước, dấu liền nhau giữ dấu mạnh nhất."""
    out: list[str] = []
    for token in tokens:
        if token in PUNCTUATION:
            if not out:
                continue
            if out[-1] in PUNCTUATION:
                if data.punctuation_strength[token] < data.punctuation_strength[out[-1]]:
                    out[-1] = token
                continue
            out.append(token)
        else:
            out += token.split()
    return "".join(t if t in PUNCTUATION or i == 0 else " " + t for i, t in enumerate(out))


def _restyle(text: str, style: str) -> str:
    """Đưa dấu thanh của vần mở oa, oe, uy về kiểu ``style`` (new: hoà, old: hòa)."""
    decomposed = unicodedata.normalize("NFD", text)
    for pattern, repl in _RESTYLE[style]:
        decomposed = pattern.sub(repl, decomposed)
    return unicodedata.normalize("NFC", decomposed)


def _context_form(text: str) -> str:
    """Dạng so khớp của từ ngữ cảnh: chữ thường, bỏ dấu kiểu mới."""
    return _restyle(unicodedata.normalize("NFC", text.lower()), "new")


# --- Đọc file dữ liệu -------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} phải là một bảng YAML")
    return raw


def _load_table(path: Path) -> dict[str, dict[str, str]]:
    """TSV ``dialect <tab> key <tab> reading [<tab> note]`` -> key -> {vùng hoặc "*": cách đọc}."""
    table: dict[str, dict[str, str]] = {}
    header_seen = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        where = f"{path.name}:{lineno}"
        if not header_seen:
            if cols[:3] != ["dialect", "key", "reading"]:
                raise ValueError(f"{where}: dòng tiêu đề phải là dialect<tab>key<tab>reading[<tab>note]")
            header_seen = True
            continue
        if len(cols) not in (3, 4):
            raise ValueError(f"{where}: cần 3 hoặc 4 cột, gặp {len(cols)}")
        dialect, key, reading = cols[0], cols[1], unicodedata.normalize("NFC", cols[2])
        if dialect != "*" and dialect not in DIALECTS:
            raise ValueError(f"{where}: vùng {dialect!r} không hợp lệ")
        if not key:
            raise ValueError(f"{where}: thiếu key")
        if not _READING.fullmatch(reading):
            raise ValueError(f"{where}: cách đọc {reading!r} phải là chữ thường tiếng Việt cách nhau một dấu cách")
        if dialect in table.get(key, {}):
            raise ValueError(f"{where}: {key!r} đã có cho vùng {dialect!r}")
        table.setdefault(key, {})[dialect] = reading
    if not header_seen:
        raise ValueError(f"{path.name}: thiếu dòng tiêu đề")
    return table


def _for_dialect(table: dict[str, dict[str, str]], dialect: str, name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, readings in table.items():
        reading = readings.get(dialect, readings.get("*"))
        if reading is None:
            raise ValueError(f"{name}.tsv: {key!r} thiếu cách đọc cho vùng {dialect!r} (hoặc dòng *)")
        out[key] = reading
    return out


def _dialect_tables(tables: dict[str, dict[str, dict[str, str]]], dialect: str) -> dict[str, Any]:
    return {name: _for_dialect(tables[name], dialect, name) for name in _TABLES}


def _word_value(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _READING.fullmatch(value):
        raise ValueError(f"{where}: {value!r} phải là chữ thường tiếng Việt")
    return value


def _int_map(raw: Any, where: str) -> dict[int, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} phải là bảng số -> chữ")
    return {int(k): _word_value(v, f"{where}.{k}") for k, v in raw.items()}


def _number_words(numbers: dict[str, Any], dialect: str) -> NumberWords:
    merged = {**numbers["common"], **(numbers["dialects"][dialect] or {})}
    where = f"numbers.yaml ({dialect})"
    digits = tuple(_word_value(d, f"{where}.digits") for d in merged["digits"])
    if len(digits) != 10:
        raise ValueError(f"{where}: digits cần đúng 10 từ")
    maps = ("after_ten", "after_tens", "ordinal", "month")
    skip = {"digits", *maps}
    fields: dict[str, Any] = {
        k: _word_value(merged[k], f"{where}.{k}") for k in NumberWords.__dataclass_fields__ if k not in skip
    }
    fields.update({k: _int_map(merged[k], f"{where}.{k}") for k in maps})
    return NumberWords(digits=digits, **fields)


def _positive_int(raw: Any, where: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        raise ValueError(f"config.yaml: {where} phải là số nguyên dương, gặp {raw!r}")
    return raw


def _build(
    config: dict[str, Any], numbers: dict[str, Any], tables: dict[str, dict[str, dict[str, str]]]
) -> NormalizeData:
    tone_style = config["tone_style"]
    if tone_style not in _RESTYLE:
        raise ValueError(f"config.yaml: tone_style phải là {' hoặc '.join(_RESTYLE)}, gặp {tone_style!r}")
    use_resolver = config["ambiguity"]["use_resolver"]
    if not isinstance(use_resolver, bool):
        raise ValueError(f"config.yaml: ambiguity.use_resolver phải là true/false, gặp {use_resolver!r}")

    punctuation = {str(k): str(v) for k, v in config["punctuation"].items()}
    strength = [str(p) for p in config["punctuation_strength"]]
    if sorted(strength) != sorted(PUNCTUATION) or any(v not in PUNCTUATION for v in punctuation.values()):
        raise ValueError(f"config.yaml: punctuation chỉ được đổi ra {PUNCTUATION}; punctuation_strength phải đủ 4 dấu")
    separator = config["phone"]["group_separator"]
    if separator not in PUNCTUATION:
        raise ValueError(f"config.yaml: phone.group_separator phải là một trong {PUNCTUATION}")
    phone_groups = {
        _positive_int(n, "phone.groups"): tuple(_positive_int(s, "phone.groups") for s in sizes)
        for n, sizes in config["phone"]["groups"].items()
    }
    if any(sum(sizes) != n for n, sizes in phone_groups.items()):
        raise ValueError("config.yaml: phone.groups: tổng các nhóm phải bằng số chữ số")

    context = {
        key: tuple(tuple(_context_form(str(p)).split()) for p in config["context"][key]) for key in _CONTEXT_KEYS
    }
    dialects = {
        d: DialectData(
            numbers=_number_words(numbers, d),
            **_dialect_tables(tables, d),
        )
        for d in DIALECTS
    }
    for d, dd in dialects.items():
        bad = [k for k in dd.letters if len(k) != 1 or not k.isupper()]
        if bad:
            raise ValueError(f"letters.tsv: key phải là một chữ hoa, gặp {bad}")
        if "+" not in dd.symbols:
            raise ValueError(f"symbols.tsv: thiếu '+' cho vùng {d} (dùng đọc mã nước +84)")
    prefixes = [str(p) for p in config["currency_prefix"]]
    missing = [p for p in prefixes if p not in tables["units"]]
    if missing:
        raise ValueError(f"config.yaml: currency_prefix {missing} chưa có cách đọc trong units.tsv")

    glued = [str(u) for u in config["unit_glued"]]
    resolution = config["resolution"]
    resolution_unit = str(resolution["unit"])
    compact = {str(k): _positive_int(v, "compact_amount") for k, v in config["compact_amount"].items()}
    unknown = [u for u in [*glued, resolution_unit] if u not in tables["units"]]
    if unknown or resolution_unit not in tables["letters"]:
        raise ValueError(f"config.yaml: unit_glued/resolution có đơn vị chưa có trong units.tsv/letters.tsv: {unknown}")
    if any(v < 1000 or v % 1000 for v in compact.values()):
        raise ValueError("config.yaml: compact_amount phải là bội của 1000")

    numbers_cfg = config["numbers"]
    return NormalizeData(
        dialects=dialects,
        use_resolver=use_resolver,
        tone_style=tone_style,
        max_digits=_positive_int(numbers_cfg["max_digits"], "numbers.max_digits"),
        max_leading_zero_digits=_positive_int(
            numbers_cfg["max_leading_zero_digits"], "numbers.max_leading_zero_digits"
        ),
        max_fraction_digits=_positive_int(numbers_cfg["max_fraction_digits"], "numbers.max_fraction_digits"),
        max_spell_letters=_positive_int(numbers_cfg["max_spell_letters"], "numbers.max_spell_letters"),
        phone_groups=phone_groups,
        phone_separator=separator,
        max_hour=_positive_int(config["time"]["max_hour"], "time.max_hour"),
        context=context,
        punctuation=punctuation,
        punctuation_strength={p: i for i, p in enumerate(strength)},
        char_map=str.maketrans({str(k): str(v) for k, v in config["char_map"].items()}),
        resolution_unit=resolution_unit,
        resolution_numbers=frozenset(_positive_int(n, "resolution.numbers") for n in resolution["numbers"]),
        compact_amount=compact,
        pattern=_build_pattern(
            abbreviations=list(tables["abbreviations"]),
            units=list(tables["units"]),
            glued=glued,
            prefixes=prefixes,
            tlds=[str(t) for t in config["url"]["tlds"]],
            compact=list(compact),
        ),
    )


def _alternation(keys: list[str], word_end: bool = False) -> str:
    """Khớp chuỗi dài trước (m² trước m, TP.HCM trước TP). Danh sách rỗng thì không khớp gì.

    ``word_end``: key kết thúc bằng chữ/số không được dính chữ phía sau (TP không khớp trong TPA);
    key kết thúc bằng dấu chấm thì được (Q.1, TP.HN).
    """
    if not keys:
        return "(?!)"
    ordered = sorted(keys, key=lambda k: (-len(k), k))
    return "|".join(re.escape(k) + (r"(?!\w)" if word_end and k[-1].isalnum() else "") for k in ordered)


def _build_pattern(
    abbreviations: list[str],
    units: list[str],
    glued: list[str],
    prefixes: list[str],
    tlds: list[str],
    compact: list[str],
) -> re.Pattern[str]:
    """Regex tổng: thứ tự nhánh là thứ tự ưu tiên khi cùng vị trí bắt đầu."""
    tld = _alternation(tlds)
    pre = _alternation(prefixes)
    spaced = _alternation([u for u in units if u not in glued])
    every = _alternation(units)

    def unit(name: str) -> str:
        # Đơn vị dính số, hoặc cách một dấu cách (trừ đơn vị trong unit_glued); không dính chữ hay "-chữ" (K-pop).
        return rf"(?:(?: (?P<{name}s>{spaced})|(?P<{name}>{every}))(?!\w|[-–][^\W\d_]))"

    def hour(p: str) -> str:
        return rf"(?P<{p}_h>[0-9]{{1,2}})(?:[hH](?P<{p}_m>[0-9]{{2}})(?:ph|p|')?|g(?P<{p}_g>[0-9]{{2}})|[hH])(?!\w)"

    def clock(p: str) -> str:
        return rf"(?P<{p}_a>[0-9]{{1,6}}):(?P<{p}_b>[0-9]{{1,6}})(?::(?P<{p}_c>[0-9]{{1,6}}))?"

    branches = [
        rf"(?P<url>https?://[^\s]*{_URL_END})",
        r"(?P<email>(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)",
        rf"(?P<domain>(?<![\w.@-])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:{tld})(?![\w-])(?:/[^\s]*{_URL_END}|/)?)",
        r"(?P<dmy>(?<![0-9.,/:-])(?P<dmy_d>[0-9]{1,2})(?P<dmy_s>[/.-])(?P<dmy_m>[0-9]{1,2})(?P=dmy_s)"
        r"(?P<dmy_y>[0-9]{4})(?![0-9]|[.,/-][0-9]))",
        r"(?P<ymd>(?<![0-9.,/:-])(?P<ymd_y>[0-9]{4})-(?P<ymd_m>[0-9]{1,2})-(?P<ymd_d>[0-9]{1,2})(?![0-9]|[.,/-][0-9]))",
        r"(?P<phone>(?<![0-9+])(?:\+84[ .-]?[0-9]{2,3}[ .-]?[0-9]{3,4}[ .-]?[0-9]{3,4}"
        r"|\(?0[0-9]{2,3}\)?[ .-]?[0-9]{3,4}[ .-]?[0-9]{3,4}|1[89]00[ .]?[0-9]{4,6})(?![0-9]|[.,][0-9]))",
        rf"(?P<timeh>(?<![0-9.,:/]){hour('th')}(?: ?[-–] ?{hour('th2')})?)",
        rf"(?P<colon>(?<![0-9.,:/]){clock('c')}(?: ?[-–] ?{clock('c2')})?(?![0-9]|:[0-9]))",
        r"(?P<slash>(?<![0-9.,:/])(?P<s_a>[0-9]{1,6})/(?P<s_b>[0-9]{1,6})(?![0-9]|/[0-9]|[.,][0-9]))",
        rf"(?P<compact>(?<![0-9.,])(?P<cm_a>[0-9]{{1,3}})(?P<cm_u>{_alternation(compact)})(?P<cm_b>[0-9]{{1,3}})(?!\w))",
        rf"(?P<quant>(?:(?P<q_pre>{pre}) ?)?(?:(?<![\w%°)\]])(?P<q_neg>[-−]))?(?<![0-9])(?P<q_a>{_NUM}){unit('q_ua')}?"
        rf"(?: ?[-–] ?(?:(?P<q_pre_b>{pre}) ?)?(?P<q_b>{_NUM}){unit('q_ub')}?)?)",
        rf"(?P<abbr>(?<!\w)(?:{_alternation(abbreviations, word_end=True)}))",
        r"(?P<hyph>(?<![\w-])[A-Za-z]+(?:-[A-Za-z]+)+(?![\w-]))",
        rf"(?P<word>{_WORD}+)",
        r"(?P<newline>\n+)",
        r"(?P<other>\S)",
    ]
    return re.compile("|".join(branches))
