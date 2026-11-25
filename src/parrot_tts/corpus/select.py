"""Chọn câu phủ âm vị và thanh điệu bằng thuật toán tham lam (đặc tả mục 5).

Mỗi bước chọn câu có lợi lớn nhất trên các mục tiêu còn thiếu:
lợi = tổng trên mọi cặp của min(số lần trong câu, số lần còn cần). Mục tiêu là các cặp (âm vị, thanh)
và cặp thanh liền kề hợp lệ của vùng, với ngưỡng trong ``config.yaml``.
Hoà lợi thì lấy câu ít âm tiết hơn, rồi câu đứng trước trong pool: cùng đầu vào cho cùng kết quả.

Lợi của một câu chỉ giảm khi số lần còn cần giảm, nên dùng tham lam lười (hàng đợi ưu tiên,
chỉ tính lại câu ở đầu hàng): cho đúng kết quả như tính lại cả pool mỗi bước nhưng nhanh hơn nhiều.

Câu trong pool là văn bản thô: qua P1 (chuẩn hoá) rồi P2 (G2P) trước khi đếm. Câu trùng nội dung sau
chuẩn hoá chỉ giữ câu đứng trước; câu bị bỏ được ghi trong ``Selection.dropped_duplicates``.
"""

import heapq
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from parrot_tts.corpus.checks import Normalizer, normalize_for_compare
from parrot_tts.corpus.coverage import (
    CorpusConfig,
    Sentence,
    count_sentence,
    estimate_seconds,
    load_config,
    valid_pairs,
    valid_tone_pairs,
)
from parrot_tts.text.g2p import g2p
from parrot_tts.text.normalize import normalize

# Mục tiêu: ("pair", âm vị, thanh) hoặc ("tone_pair", thanh trước, thanh sau).
Target = tuple[str, str, str]


class StopReason(StrEnum):
    TARGETS_MET = "đạt mọi mục tiêu"
    POOL_EXHAUSTED = "hết câu trong pool"
    NO_GAIN = "không câu nào còn thêm được cặp thiếu"
    MAX_SENTENCES = "đủ số câu tối đa"
    MAX_SECONDS = "đủ thời lượng tối đa"


@dataclass(frozen=True)
class Unit[K: Hashable]:
    """Một câu ứng viên đã đếm: mục tiêu -> số lần xuất hiện trong câu."""

    id: str
    counts: Mapping[K, int]
    syllables: int
    seconds: float


@dataclass(frozen=True)
class Selection[K: Hashable]:
    ids: tuple[str, ...]  # theo thứ tự chọn
    stop_reason: StopReason
    remaining: dict[K, int]  # mục tiêu còn thiếu -> số lần còn cần
    seconds: float  # tổng thời lượng ước tính của các câu đã chọn
    dropped_duplicates: tuple[tuple[str, str], ...] = ()  # (mã bị bỏ, mã câu trùng được giữ)


def greedy_pick[K: Hashable](
    units: Sequence[Unit[K]],
    need: Mapping[K, int],
    *,
    max_sentences: int | None = None,
    max_seconds: float | None = None,
) -> Selection[K]:
    """Chọn tham lam từ ``units`` cho tới khi đạt ``need``, hết pool, hết lợi hoặc chạm ngân sách."""
    if max_sentences is not None and max_sentences < 0:
        raise ValueError(f"max_sentences không được âm, gặp {max_sentences}")
    if max_seconds is not None and max_seconds < 0:
        raise ValueError(f"max_seconds không được âm, gặp {max_seconds}")
    duplicated = _duplicates(u.id for u in units)
    if duplicated:
        raise ValueError(f"trùng mã câu trong pool: {', '.join(duplicated)}")

    remaining = {k: n for k, n in need.items() if n > 0}

    def gain(u: Unit[K]) -> int:
        return sum(min(c, remaining[k]) for k, c in u.counts.items() if k in remaining)

    # Khoá nhỏ nhất là câu tốt nhất: lợi lớn, ít âm tiết, đứng trước.
    heap = [(-gain(u), u.syllables, i) for i, u in enumerate(units)]
    heapq.heapify(heap)
    chosen: list[str] = []
    seconds = 0.0
    skipped_for_time = False

    while True:
        if not remaining:
            reason = StopReason.TARGETS_MET
            break
        if max_sentences is not None and len(chosen) >= max_sentences:
            reason = StopReason.MAX_SENTENCES
            break
        if not heap:
            reason = StopReason.MAX_SECONDS if skipped_for_time else StopReason.POOL_EXHAUSTED
            break
        _, syllables, index = heapq.heappop(heap)
        unit = units[index]
        if max_seconds is not None and seconds + unit.seconds > max_seconds:
            # Thời lượng còn lại chỉ giảm: câu này không bao giờ vừa nữa. Ghi lại nếu nó còn có ích.
            skipped_for_time = skipped_for_time or gain(unit) > 0
            continue
        fresh = (-gain(unit), syllables, index)
        if heap and fresh > heap[0]:
            heapq.heappush(heap, fresh)  # lợi cũ đã lỗi thời, xếp lại
            continue
        if fresh[0] == 0:
            reason = StopReason.MAX_SECONDS if skipped_for_time else StopReason.NO_GAIN
            break
        chosen.append(unit.id)
        seconds += unit.seconds
        for key, count in unit.counts.items():
            if key in remaining:
                remaining[key] -= count
                if remaining[key] <= 0:
                    del remaining[key]

    return Selection(ids=tuple(chosen), stop_reason=reason, remaining=remaining, seconds=seconds)


def coverage_need(
    dialect: str,
    already: Iterable[str] = (),
    config: CorpusConfig | None = None,
    *,
    normalizer: Normalizer = normalize,
) -> dict[Target, int]:
    """Số lần còn cần cho từng mục tiêu, sau khi trừ các câu thô ``already`` đã có trong kịch bản."""
    config = config or load_config()
    need: dict[Target, int] = {("pair", *p): config.min_pair_count for p in sorted(valid_pairs(dialect))}
    need |= {("tone_pair", *p): config.min_tone_pair_count for p in sorted(valid_tone_pairs(dialect))}
    for text in already:
        for key, count in _unit(Sentence(id="", text=normalizer(text, dialect)), dialect, config).counts.items():
            if key in need:
                need[key] = max(0, need[key] - count)
    return need


def select_sentences(
    pool: Sequence[Sentence],
    dialect: str,
    *,
    max_sentences: int | None = None,
    max_seconds: float | None = None,
    already: Iterable[str] = (),
    config: CorpusConfig | None = None,
    normalizer: Normalizer = normalize,
) -> Selection[Target]:
    """Chọn tham lam trên câu thô. Mọi câu lỗi chuẩn hoá hoặc G2P được gom lại rồi ném một lần."""
    config = config or load_config()
    g2p("", dialect)  # vùng sai: ném ngay
    units: list[Unit[Target]] = []
    errors: list[str] = []
    kept: dict[str, str] = {}  # chữ so khớp -> mã câu giữ lại
    dropped: list[tuple[str, str]] = []
    for sentence in pool:
        try:
            normalized = replace(sentence, text=normalizer(sentence.text, dialect))
            unit = _unit(normalized, dialect, config)
        except ValueError as err:
            errors.append(f"{sentence.id}: {err}")
            continue
        key = normalize_for_compare(normalized.text)
        if key in kept:
            dropped.append((sentence.id, kept[key]))
            continue
        kept[key] = sentence.id
        units.append(unit)
    if errors:
        raise ValueError(f"{len(errors)} câu không qua chuẩn hoá hoặc G2P:\n" + "\n".join(errors))
    need = coverage_need(dialect, already, config, normalizer=normalizer)
    result = greedy_pick(units, need, max_sentences=max_sentences, max_seconds=max_seconds)
    return replace(result, dropped_duplicates=tuple(dropped))


def category_budget(total: int, config: CorpusConfig | None = None) -> dict[str, int]:
    """Chia ``total`` câu theo tỉ lệ loại câu (đặc tả mục 5), làm tròn theo phần dư lớn nhất.

    Hoà phần dư thì loại đứng trước trong ``config.yaml`` được thêm trước. Tổng luôn bằng ``total``.
    """
    if total < 0:
        raise ValueError(f"tổng số câu không được âm, gặp {total}")
    quotas = (config or load_config()).quotas
    names = list(quotas)
    exact = [quotas[name] * total for name in names]
    budget = [int(value) for value in exact]
    left = total - sum(budget)
    for i in sorted(range(len(names)), key=lambda i: (-(exact[i] - budget[i]), i))[:left]:
        budget[i] += 1
    return dict(zip(names, budget, strict=True))


def select_coverage_share(
    pool: Sequence[Sentence],
    dialect: str,
    total_sentences: int,
    *,
    total_seconds: float | None = None,
    already: Iterable[str] = (),
    config: CorpusConfig | None = None,
    normalizer: Normalizer = normalize,
) -> Selection[Target]:
    """Chọn phần "câu phủ âm vị và thanh điệu" của kịch bản: chỉ từ câu thuộc loại đó trong pool,
    tối đa đúng số câu ``category_budget`` chia cho loại này (và tỉ lệ đó của ``total_seconds`` nếu có).
    """
    config = config or load_config()
    category = config.coverage_category
    max_seconds = None if total_seconds is None else total_seconds * config.quotas[category]
    return select_sentences(
        [s for s in pool if s.category == category],
        dialect,
        max_sentences=category_budget(total_sentences, config)[category],
        max_seconds=max_seconds,
        already=already,
        config=config,
        normalizer=normalizer,
    )


def _unit(sentence: Sentence, dialect: str, config: CorpusConfig) -> Unit[Target]:
    counts = count_sentence(sentence.text, dialect, config)
    targets: dict[Target, int] = {("pair", *p): n for p, n in counts.pairs.items()}
    targets |= {("tone_pair", *p): n for p, n in counts.tone_pairs.items()}
    return Unit(
        id=sentence.id,
        counts=targets,
        syllables=counts.syllables,
        seconds=estimate_seconds(counts.syllables, config),
    )


def _duplicates(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dup: list[str] = []
    for i in ids:
        if i in seen and i not in dup:
            dup.append(i)
        seen.add(i)
    return dup
