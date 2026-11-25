"""Đo độ phủ của một bộ câu: cặp (âm vị, thanh) và cặp thanh liền kề (plan P3, đặc tả mục 5).

Cặp (âm vị, thanh): mỗi token âm vị của một âm tiết (kể cả ``ʔ``) đi với token thanh của âm tiết đó.
Cặp thanh liền kề: thanh của hai âm tiết liên tiếp trong câu, qua cả ``_``. Mặc định dấu , . ? !
cắt chuỗi (``tone_pairs_across_punctuation`` trong ``config.yaml``), vì người đọc thường ngừng ở đó.

Tập cặp hợp lệ của một vùng được liệt kê từ chính bảng chữ viết của G2P: ghép mọi âm đầu × vần ×
âm cuối × 6 dấu thanh, giữ các âm tiết G2P đọc được. Nhờ vậy âm cuối tắc (k͡p, ʲk...) chỉ đi với T7, T8,
và giọng Nam (ngã nhập hỏi) không có T3, nên chỉ có 7 × 7 = 49 cặp thanh liền kề thay vì 64.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from vietvoice.text import g2p as g2p_module
from vietvoice.text.g2p import PUNCTUATION, WORD_BOUNDARY, g2p

CONFIG_PATH = Path(__file__).parent / "config.yaml"
DIALECT_NAMES = {"bac": "Bắc", "nam": "Nam"}
ALL_TONES = tuple(f"T{n}" for n in range(1, 9))

_TONE_TOKEN = re.compile(r"^T[1-8]$")

Pair = tuple[str, str]


@dataclass(frozen=True)
class Sentence:
    """Một câu đã chuẩn hoá (không còn chữ số), kèm mã và loại câu."""

    id: str
    text: str
    category: str = ""


@dataclass(frozen=True)
class CorpusConfig:
    """Ngưỡng đọc từ ``config.yaml``."""

    min_pair_count: int
    min_tone_pair_count: int
    tone_pairs_across_punctuation: bool
    report_max_listed: int
    speech_rate: float  # âm tiết/giây
    min_seconds: float
    max_seconds: float
    quotas: dict[str, float]  # loại câu -> tỉ lệ, giữ thứ tự trong file
    coverage_category: str


@dataclass(frozen=True)
class SentenceCounts:
    """Số đếm của một câu."""

    pairs: Counter[Pair]
    tone_pairs: Counter[Pair]
    syllables: int


@dataclass(frozen=True)
class CoverageReport:
    """Độ phủ của một bộ câu theo một vùng."""

    dialect: str
    sentences: int
    syllables: int
    pair_counts: Counter[Pair]
    tone_pair_counts: Counter[Pair]
    valid_pairs: frozenset[Pair]
    valid_tone_pairs: frozenset[Pair]
    tones: tuple[str, ...]  # các thanh vùng này có
    min_pair_count: int
    min_tone_pair_count: int

    @property
    def missing_pairs(self) -> dict[Pair, int]:
        """Cặp (âm vị, thanh) hợp lệ còn dưới ngưỡng -> số lần hiện có."""
        return _below(self.valid_pairs, self.pair_counts, self.min_pair_count)

    @property
    def missing_tone_pairs(self) -> dict[Pair, int]:
        """Cặp thanh liền kề hợp lệ còn dưới ngưỡng -> số lần hiện có."""
        return _below(self.valid_tone_pairs, self.tone_pair_counts, self.min_tone_pair_count)

    @property
    def unexpected_pairs(self) -> dict[Pair, int]:
        """Cặp gặp trong câu nhưng không có trong tập hợp lệ: dấu hiệu luật và bảng liệt kê lệch nhau."""
        return {p: n for p, n in sorted(self.pair_counts.items()) if p not in self.valid_pairs}

    @property
    def complete(self) -> bool:
        return not self.missing_pairs and not self.missing_tone_pairs


def load_config(path: Path | None = None) -> CorpusConfig:
    """Đọc ``config.yaml`` (mặc định file cạnh module này) và kiểm tra giá trị."""
    return _load_config(path or CONFIG_PATH)


def pairs_in_tokens(tokens: Sequence[str]) -> Counter[Pair]:
    """Đếm cặp (âm vị, thanh) trong chuỗi token G2P."""
    counts: Counter[Pair] = Counter()
    for phonemes, tone in _syllables(tokens):
        counts.update((p, tone) for p in phonemes)
    return counts


def tone_pairs_in_tokens(tokens: Sequence[str], across_punctuation: bool = False) -> Counter[Pair]:
    """Đếm cặp thanh của hai âm tiết liên tiếp; dấu câu cắt chuỗi trừ khi ``across_punctuation``."""
    counts: Counter[Pair] = Counter()
    previous: str | None = None
    for token in tokens:
        if token in PUNCTUATION:
            if not across_punctuation:
                previous = None
        elif _TONE_TOKEN.match(token):
            if previous is not None:
                counts[(previous, token)] += 1
            previous = token
    return counts


def count_sentence(text: str, dialect: str, config: CorpusConfig | None = None) -> SentenceCounts:
    """G2P một câu rồi đếm. Lỗi G2P được ném tiếp nguyên vẹn."""
    config = config or load_config()
    tokens = g2p(text, dialect)
    return SentenceCounts(
        pairs=pairs_in_tokens(tokens),
        tone_pairs=tone_pairs_in_tokens(tokens, config.tone_pairs_across_punctuation),
        syllables=sum(1 for t in tokens if _TONE_TOKEN.match(t)),
    )


def estimate_seconds(syllables: int, config: CorpusConfig) -> float:
    """Thời lượng đọc ước tính = số âm tiết / tốc độ nói."""
    return syllables / config.speech_rate


@cache
def valid_pairs(dialect: str) -> frozenset[Pair]:
    """Mọi cặp (âm vị, thanh) mà luật của vùng sinh ra được."""
    g2p("", dialect)  # nạp luật trước: vùng sai hay file luật hỏng thì ném lỗi ngay, không bị lọc mất
    pairs: set[Pair] = set()
    for word in _candidate_syllables():
        try:
            tokens = g2p(word, dialect)
        except ValueError:
            continue  # tổ hợp chữ viết không thành âm tiết: bị loại có chủ đích
        pairs.update(pairs_in_tokens(tokens))
    if not pairs:
        raise ValueError(f"không liệt kê được âm tiết nào cho vùng {dialect!r}")
    return frozenset(pairs)


def valid_tones(dialect: str) -> tuple[str, ...]:
    """Các thanh vùng này có, theo thứ tự T1..T8."""
    present = {tone for _, tone in valid_pairs(dialect)}
    return tuple(t for t in ALL_TONES if t in present)


def valid_tone_pairs(dialect: str) -> frozenset[Pair]:
    """Mọi cặp thanh liền kề của vùng: số thanh × số thanh."""
    tones = valid_tones(dialect)
    return frozenset((a, b) for a in tones for b in tones)


def measure(texts: Iterable[str], dialect: str, config: CorpusConfig | None = None) -> CoverageReport:
    """Đo độ phủ của các câu theo ``dialect``. Câu lỗi G2P làm hàm ném lỗi."""
    config = config or load_config()
    pairs: Counter[Pair] = Counter()
    tone_pairs: Counter[Pair] = Counter()
    sentences = syllables = 0
    for text in texts:
        counts = count_sentence(text, dialect, config)
        pairs.update(counts.pairs)
        tone_pairs.update(counts.tone_pairs)
        syllables += counts.syllables
        sentences += 1
    return CoverageReport(
        dialect=dialect,
        sentences=sentences,
        syllables=syllables,
        pair_counts=pairs,
        tone_pair_counts=tone_pairs,
        valid_pairs=valid_pairs(dialect),
        valid_tone_pairs=valid_tone_pairs(dialect),
        tones=valid_tones(dialect),
        min_pair_count=config.min_pair_count,
        min_tone_pair_count=config.min_tone_pair_count,
    )


def render_report(report: CoverageReport, max_listed: int | None = None) -> str:
    """Báo cáo ngắn bằng tiếng Việt."""
    if max_listed is None:
        max_listed = load_config().report_max_listed
    name = DIALECT_NAMES.get(report.dialect, report.dialect)
    n_tones = len(report.tones)
    absent = [t for t in ALL_TONES if t not in report.tones]
    missing = report.missing_pairs
    missing_tone = report.missing_tone_pairs

    lines = [
        f"Độ phủ giọng {name}: {report.sentences} câu, {report.syllables} âm tiết.",
        f"Cặp (âm vị, thanh) hợp lệ: {len(report.valid_pairs)}; "
        f"đạt ≥ {report.min_pair_count} lần: {len(report.valid_pairs) - len(missing)}; còn thiếu: {len(missing)}.",
    ]
    if missing:
        lines.append("  Thiếu nhiều nhất: " + _list_counts(missing, max_listed))
    tone_note = f"{n_tones} thanh × {n_tones} = {len(report.valid_tone_pairs)} cặp"
    if absent:
        tone_note += f"; giọng {name} không có {', '.join(absent)} nên không tính cặp có thanh đó"
    lines.append(
        f"Cặp thanh liền kề ({tone_note}): đạt ≥ {report.min_tone_pair_count} lần: "
        f"{len(report.valid_tone_pairs) - len(missing_tone)}; còn thiếu: {len(missing_tone)}."
    )
    if missing_tone:
        lines.append("  Thiếu: " + _list_counts(missing_tone, max_listed))
    unexpected = report.unexpected_pairs
    if unexpected:
        lines.append(
            "Cảnh báo: cặp không có trong tập hợp lệ (luật G2P và bảng liệt kê lệch nhau): "
            + _list_counts(unexpected, max_listed)
        )
    lines.append("Kết luận: " + ("đủ độ phủ." if report.complete else "chưa đủ độ phủ."))
    return "\n".join(lines)


def _syllables(tokens: Sequence[str]) -> list[tuple[list[str], str]]:
    """Tách token thành (âm vị, thanh) cho từng âm tiết."""
    out: list[tuple[list[str], str]] = []
    current: list[str] = []
    for token in tokens:
        if token in PUNCTUATION or token == WORD_BOUNDARY:
            if current:
                raise ValueError(f"âm tiết thiếu token thanh ở cuối: {current}")
            continue
        if _TONE_TOKEN.match(token):
            out.append((current, token))
            current = []
        else:
            current.append(token)
    if current:
        raise ValueError(f"âm tiết thiếu token thanh ở cuối: {current}")
    return out


@cache
def _candidate_syllables() -> tuple[str, ...]:
    """Mọi tổ hợp âm đầu × vần × âm cuối × dấu thanh từ bảng chữ viết của G2P (có cả tổ hợp sai)."""
    onsets = ("", *g2p_module._ONSETS)
    codas = ("", *g2p_module._CONSONANT_CODAS)
    marks = ("", *g2p_module._TONE_MARKS)
    words = []
    for onset in onsets:
        for vowels in g2p_module._VOWELS:
            for coda in codas:
                for mark in marks:
                    # Dấu đặt sau chữ nguyên âm đầu; G2P chỉ cần có dấu, không cần đúng vị trí.
                    word = onset + vowels[0] + mark + vowels[1:] + coda
                    words.append(unicodedata.normalize("NFC", word))
    return tuple(words)


def _below(valid: frozenset[Pair], counts: Counter[Pair], minimum: int) -> dict[Pair, int]:
    return {p: counts[p] for p in sorted(valid) if counts[p] < minimum}


def _list_counts(items: dict[Pair, int], limit: int) -> str:
    ordered = sorted(items.items(), key=lambda kv: (kv[1], kv[0]))
    shown = ", ".join(f"({a}, {b})={n}" for (a, b), n in ordered[:limit])
    rest = len(ordered) - limit
    return shown + (f", … và {rest} cặp nữa" if rest > 0 else "")


@cache
def _load_config(path: Path) -> CorpusConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        coverage, duration, script = raw["coverage"], raw["duration"], raw["script"]
        config = CorpusConfig(
            min_pair_count=_int(coverage["min_pair_count"]),
            min_tone_pair_count=_int(coverage["min_tone_pair_count"]),
            tone_pairs_across_punctuation=_bool(coverage["tone_pairs_across_punctuation"]),
            report_max_listed=_int(coverage["report_max_listed"]),
            speech_rate=_float(duration["speech_rate"]),
            min_seconds=_float(duration["min_seconds"]),
            max_seconds=_float(duration["max_seconds"]),
            quotas={str(k): _float(v) for k, v in script["quotas"].items()},
            coverage_category=str(script["coverage_category"]),
        )
    except (KeyError, TypeError, AttributeError) as err:
        raise ValueError(f"{path.name} sai cấu trúc: {err!r}") from err

    if min(config.min_pair_count, config.min_tone_pair_count, config.report_max_listed) < 0:
        raise ValueError(f"{path.name}: ngưỡng đếm không được âm")
    if config.speech_rate <= 0:
        raise ValueError(f"{path.name}: speech_rate phải dương, gặp {config.speech_rate}")
    if not 0 <= config.min_seconds < config.max_seconds:
        raise ValueError(f"{path.name}: cần 0 ≤ min_seconds < max_seconds")
    if any(v < 0 for v in config.quotas.values()):
        raise ValueError(f"{path.name}: tỉ lệ loại câu không được âm")
    if abs(sum(config.quotas.values()) - 1) > 1e-9:
        raise ValueError(f"{path.name}: tổng tỉ lệ loại câu phải bằng 1, gặp {sum(config.quotas.values())}")
    if config.coverage_category not in config.quotas:
        raise ValueError(f"{path.name}: coverage_category {config.coverage_category!r} không có trong quotas")
    return config


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"cần số nguyên, gặp {value!r}")
    return value


def _float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"cần số, gặp {value!r}")
    return float(value)


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"cần true/false, gặp {value!r}")
    return value
