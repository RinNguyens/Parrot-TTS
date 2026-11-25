"""G2P tiếng Việt theo vùng: chữ viết -> chuỗi âm vị và token thanh (đặc tả mục 3).

Code ở đây chỉ tách chữ viết thành âm đầu, âm đệm, âm chính, âm cuối và thanh.
Mỗi phần đọc ra sao ở từng vùng nằm trong ``rules/<vùng>.yaml``, do chủ dự án viết và duyệt.

Từ nước ngoài và tên riêng tra ở ``lexicon/<vùng>.tsv`` trước, rồi cách đọc ghi trong đó
(âm tiết Việt, ví dụ tiktok -> "tích tốc") mới đọc bằng luật của vùng.

Văn bản đầu vào phải đã qua chuẩn hoá (số, viết tắt, ký hiệu -> chữ). Gặp chữ không
phân tích được thì ném ``ValueError`` nêu rõ âm tiết lỗi, không bỏ qua im lặng.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

Dialect = Literal["bac", "nam"]
DIALECTS: tuple[str, ...] = get_args(Dialect)
RULES_DIR = Path(__file__).parent / "rules"
LEXICON_DIR = Path(__file__).parent / "lexicon"

# Phiên bản bộ âm vị. Thêm, bớt hay đổi ký hiệu âm vị/token trong rules/*.yaml hoặc ở đây
# thì phải tăng số này, vì mô hình đã huấn luyện gắn với đúng bộ âm vị lúc đó.
INVENTORY_VERSION = 1

PUNCTUATION = frozenset(",.?!")
WORD_BOUNDARY = "_"
GLOTTAL_STOP = "ʔ"

# Dấu thanh ở dạng Unicode tách (NFD) -> số thanh theo chữ viết. Không dấu là thanh 1 (ngang).
_TONE_MARKS = {"̀": 2, "̃": 3, "̉": 4, "́": 5, "̣": 6}
_LETTERS = frozenset("abcdeghiklmnopqrstuvxyăâđêôơư")
_VOWEL_LETTERS = frozenset("aăâeêioôơuưy")

_ONSETS = ("ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr")
_ONSETS += tuple("bcdđghklmnprstvx")
_CONSONANT_CODAS = ("ch", "ng", "nh", "c", "m", "n", "p", "t")
_CHECKED_CODAS = frozenset({"c", "ch", "p", "t"})

# Cụm nguyên âm chữ viết (đã bỏ âm đầu và âm cuối phụ âm) -> (có âm đệm w, âm chính, bán âm cuối).
# Âm chính được gộp về một cách viết: ia, yê -> iê ; ua -> uô ; ưa -> ươ ; y -> i.
# Bán âm cuối: "i" (đọc j) hoặc "u" (đọc w); tên chữ viết, cách đọc nằm trong bảng coda của YAML.
_VOWELS: dict[str, tuple[bool, str, str]] = {
    "a": (False, "a", ""),
    "ă": (False, "ă", ""),
    "â": (False, "â", ""),
    "e": (False, "e", ""),
    "ê": (False, "ê", ""),
    "i": (False, "i", ""),
    "y": (False, "i", ""),
    "o": (False, "o", ""),
    "oo": (False, "oo", ""),
    "ô": (False, "ô", ""),
    "ơ": (False, "ơ", ""),
    "u": (False, "u", ""),
    "ư": (False, "ư", ""),
    "ia": (False, "iê", ""),
    "iê": (False, "iê", ""),
    "yê": (False, "iê", ""),
    "ua": (False, "uô", ""),
    "uô": (False, "uô", ""),
    "ưa": (False, "ươ", ""),
    "ươ": (False, "ươ", ""),
    "oa": (True, "a", ""),
    "oă": (True, "ă", ""),
    "oe": (True, "e", ""),
    "uâ": (True, "â", ""),
    "uê": (True, "ê", ""),
    "uơ": (True, "ơ", ""),
    "uy": (True, "i", ""),
    "uya": (True, "iê", ""),
    "uyê": (True, "iê", ""),
    "ai": (False, "a", "i"),
    "ay": (False, "ă", "i"),
    "ây": (False, "â", "i"),
    "oi": (False, "o", "i"),
    "ôi": (False, "ô", "i"),
    "ơi": (False, "ơ", "i"),
    "ui": (False, "u", "i"),
    "ưi": (False, "ư", "i"),
    "uôi": (False, "uô", "i"),
    "ươi": (False, "ươ", "i"),
    "oai": (True, "a", "i"),
    "oay": (True, "ă", "i"),
    "uây": (True, "â", "i"),
    "ao": (False, "a", "u"),
    "au": (False, "ă", "u"),
    "âu": (False, "â", "u"),
    "eo": (False, "e", "u"),
    "êu": (False, "ê", "u"),
    "iu": (False, "i", "u"),
    "ưu": (False, "ư", "u"),
    "iêu": (False, "iê", "u"),
    "yêu": (False, "iê", "u"),
    "ươu": (False, "ươ", "u"),
    "oeo": (True, "e", "u"),
    "uyu": (True, "i", "u"),
}

_WORD = re.compile(r"[^\W\d_]+")
_PIECE = re.compile(r"[^\W\d_]+|\S")

_NUCLEI = frozenset(n for _, n, _ in _VOWELS.values())
_CODAS = frozenset(_CONSONANT_CODAS) | {"", "i", "u"}


@dataclass(frozen=True)
class Syllable:
    """Một âm tiết đã đọc theo vùng. Chuỗi rỗng là không có thành phần đó."""

    onset: str
    glide: str
    nucleus: str
    coda: str
    tone: int  # 1..8, đã áp luật vùng
    glottal_onset: bool  # âm tiết mở đầu bằng nguyên âm, đọc có ʔ


@dataclass(frozen=True)
class _Spelling:
    """Âm tiết tách theo chữ viết, chưa áp luật vùng."""

    onset: str  # "", "b", "qu", "ngh", ...
    glide: bool
    nucleus: str  # một khoá của bảng nucleus trong YAML
    coda: str  # một khoá của bảng coda trong YAML
    tone: int  # 1..8 theo chữ viết


@dataclass(frozen=True)
class _CodaRule:
    coda: str
    after: frozenset[str]
    to: str | None
    nucleus: str | None


@dataclass(frozen=True)
class _Rules:
    onset: dict[str, str]
    qu_onset: str
    qu_glide: str
    nucleus: dict[str, str]
    coda: dict[str, str]
    coda_rules: tuple[_CodaRule, ...]
    tone: dict[int, int]


def syllable_tone(syllable: str) -> int:
    """Số thanh 1..8 theo chữ viết (chưa áp luật vùng); 7, 8 là sắc, nặng ở âm tiết tắc."""
    return _spell(syllable).tone


def parse_syllable(word: str, dialect: str) -> Syllable:
    """Đọc một âm tiết theo luật của ``dialect``."""
    rules = _load_rules(dialect)
    spelling = _spell(word)

    if spelling.onset == "qu":
        onset, glide = rules.qu_onset, rules.qu_glide
    else:
        onset, glide = rules.onset[spelling.onset], "w" if spelling.glide else ""
    nucleus = rules.nucleus[spelling.nucleus]
    coda = rules.coda[spelling.coda]
    for rule in rules.coda_rules:
        if rule.coda == spelling.coda and spelling.nucleus in rule.after:
            coda = rule.to if rule.to is not None else coda
            nucleus = rule.nucleus if rule.nucleus is not None else nucleus
            break

    return Syllable(
        onset=onset,
        glide=glide,
        nucleus=nucleus,
        coda=coda,
        tone=rules.tone.get(spelling.tone, spelling.tone),
        glottal_onset=spelling.onset == "",
    )


def g2p(text: str, dialect: str) -> list[str]:
    """Văn bản đã chuẩn hoá -> token: âm vị, ``T1``..``T8`` cuối mỗi âm tiết, ``_`` giữa các âm tiết."""
    load_lexicon(dialect)
    tokens: list[str] = []
    after_syllable = False
    for piece in split_text(text):
        piece_tokens = word_tokens(piece, dialect)
        if piece in PUNCTUATION:
            after_syllable = False
        else:
            if after_syllable:
                tokens.append(WORD_BOUNDARY)
            after_syllable = True
        tokens.extend(piece_tokens)
    return tokens


def split_text(text: str) -> list[str]:
    """Tách văn bản (đưa về NFC) thành từ và từng ký tự khác khoảng trắng."""
    return _PIECE.findall(unicodedata.normalize("NFC", text))


def word_tokens(piece: str, dialect: str) -> list[str]:
    """Token của một mảnh do ``split_text`` tách ra: dấu câu, từ trong từ điển ngoại lệ, hoặc một âm tiết."""
    if piece in PUNCTUATION:
        return [piece]
    if not piece.isalpha():
        raise ValueError(f"ký tự {piece!r} chưa đọc được; văn bản cần qua bước chuẩn hoá trước G2P")
    respelling = load_lexicon(dialect).get(piece.lower(), (piece,))
    tokens: list[str] = []
    for i, syllable in enumerate(respelling):
        if i:
            tokens.append(WORD_BOUNDARY)
        tokens.extend(_to_tokens(parse_syllable(syllable, dialect)))
    return tokens


def load_lexicon(dialect: str, directory: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Từ điển ngoại lệ của vùng: từ (chữ thường, NFC) -> các âm tiết của cách đọc.

    Mặc định đọc ``LEXICON_DIR/<vùng>.tsv``. File sai thì ném ``ValueError`` nêu tên file và dòng.
    """
    _load_rules(dialect)
    return _load_lexicon(dialect, LEXICON_DIR if directory is None else directory)


def _to_tokens(syllable: Syllable) -> list[str]:
    parts = [GLOTTAL_STOP] if syllable.glottal_onset else []
    parts += [p for p in (syllable.onset, syllable.glide, syllable.nucleus, syllable.coda) if p]
    return [*parts, f"T{syllable.tone}"]


def _spell(word: str) -> _Spelling:
    base, tone = _strip_tone(word)
    if not base or not set(base) <= _LETTERS:
        raise ValueError(f"không phải âm tiết tiếng Việt: {word!r}")

    onset = next((o for o in _ONSETS if base.startswith(o)), "")
    rest = base[len(onset) :]
    if onset == "gi" and (not rest or rest[0] not in _VOWEL_LETTERS or rest[0] == "ê"):
        rest = "i" + rest  # gì, gìn, giếng: chữ i vừa thuộc gi vừa là nguyên âm

    coda = next((c for c in _CONSONANT_CODAS if rest.endswith(c)), "")
    vowels = rest[: len(rest) - len(coda)]
    if vowels not in _VOWELS:
        raise ValueError(f"không tách được vần của âm tiết {word!r}")
    glide, nucleus, semivowel = _VOWELS[vowels]
    if semivowel and coda:
        raise ValueError(f"âm tiết {word!r} có hai âm cuối")
    if onset == "qu" and glide:
        raise ValueError(f"âm tiết {word!r} có hai âm đệm")
    coda = coda or semivowel

    if coda in _CHECKED_CODAS:
        if tone not in (5, 6):
            raise ValueError(f"âm tiết tắc {word!r} chỉ mang được thanh sắc hoặc nặng")
        tone += 2
    return _Spelling(onset=onset, glide=glide, nucleus=nucleus, coda=coda, tone=tone)


def _strip_tone(word: str) -> tuple[str, int]:
    """Bỏ dấu thanh, trả về (chữ thường dạng NFC, số thanh 1..6). Chấp nhận cả NFC lẫn NFD."""
    tone = 1
    letters = []
    for char in unicodedata.normalize("NFD", word.lower()):
        if char in _TONE_MARKS:
            if tone != 1:
                raise ValueError(f"âm tiết {word!r} có hai dấu thanh")
            tone = _TONE_MARKS[char]
        else:
            letters.append(char)
    return unicodedata.normalize("NFC", "".join(letters)), tone


@cache
def _load_rules(dialect: str) -> _Rules:
    if dialect not in DIALECTS:
        raise ValueError(f"vùng {dialect!r} không hợp lệ; chỉ có: {', '.join(DIALECTS)}")
    path = RULES_DIR / f"{dialect}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        rules = _Rules(
            onset=_str_table(raw["onset"]),
            qu_onset=_str(raw["qu"]["onset"]),
            qu_glide=_str(raw["qu"]["glide"]),
            nucleus=_str_table(raw["nucleus"]),
            coda=_str_table(raw["coda"]),
            coda_rules=tuple(_coda_rule(r) for r in raw["coda_rules"]),
            tone={int(k): int(v) for k, v in (raw["tone"] or {}).items()},
        )
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"{path.name} sai cấu trúc: {err!r}") from err

    missing = {
        "onset": (set(_ONSETS) - {"qu"} | {""}) - rules.onset.keys(),
        "nucleus": _NUCLEI - rules.nucleus.keys(),
        "coda": _CODAS - rules.coda.keys(),
    }
    for table, keys in missing.items():
        if keys:
            raise ValueError(f"{path.name} thiếu {table}: {', '.join(sorted(keys))}")
    bad_tones = {t for pair in rules.tone.items() for t in pair if not 1 <= t <= 8}
    if bad_tones:
        raise ValueError(f"{path.name} có số thanh ngoài 1..8: {sorted(bad_tones)}")
    return rules


@cache
def _load_lexicon(dialect: str, directory: Path) -> dict[str, tuple[str, ...]]:
    path = directory / f"{dialect}.tsv"
    if not path.is_file():
        raise ValueError(f"không có từ điển ngoại lệ {path}")
    entries: dict[str, tuple[str, ...]] = {}
    first_line: dict[str, int] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        where = f"{path.name} dòng {number}"
        columns = line.split("\t")
        if len(columns) not in (2, 3):
            raise ValueError(f"{where}: cần 2 hoặc 3 cột cách nhau bằng TAB (từ, cách đọc, ghi chú), gặp {line!r}")
        word = unicodedata.normalize("NFC", columns[0].strip()).lower()
        syllables = tuple(unicodedata.normalize("NFC", columns[1]).split())
        if not _WORD.fullmatch(word):
            raise ValueError(f"{where}: từ {columns[0]!r} phải là một từ chỉ gồm chữ cái, không dấu cách, không số")
        if not syllables:
            raise ValueError(f"{where}: từ {word!r} chưa có cách đọc")
        for syllable in syllables:
            try:
                parse_syllable(syllable, dialect)
            except ValueError as err:
                raise ValueError(f"{where}: cách đọc {columns[1]!r} của {word!r} không đọc được: {err}") from err
        if word in entries:
            raise ValueError(f"{where}: từ {word!r} bị lặp, đã có ở dòng {first_line[word]}")
        entries[word] = syllables
        first_line[word] = number
    return entries


def _str(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"cần chuỗi, gặp {value!r}")
    return value


def _str_table(raw: Any) -> dict[str, str]:
    return {_str(k): _str(v) for k, v in raw.items()}


def _coda_rule(raw: Any) -> _CodaRule:
    to, nucleus = raw.get("to"), raw.get("nucleus")
    return _CodaRule(
        coda=_str(raw["coda"]),
        after=frozenset(_str(a) for a in raw["after"]),
        to=None if to is None else _str(to),
        nucleus=None if nucleus is None else _str(nucleus),
    )
