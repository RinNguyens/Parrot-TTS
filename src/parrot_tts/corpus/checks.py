"""Kiểm tra bộ câu test và kịch bản thu (plan P3).

Mỗi hàm trả về danh sách vấn đề kèm mã câu; danh sách rỗng là đạt. Không hàm nào dừng ở lỗi đầu tiên.
So khớp câu sau khi chuẩn hoá: NFC, chữ thường, bỏ dấu câu, gộp khoảng trắng.

``run_checks`` cho mỗi câu qua P1 (``normalize``) rồi P2 (``g2p``), đúng thứ tự của pipeline.
Câu trùng và câu lọt giữa hai bộ được so cả trên chữ thô lẫn chữ đã chuẩn hoá ("50%" khớp
"năm mươi phần trăm"); trùng ở chỗ nào cũng báo, mỗi nhóm một lần.
"""

import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from parrot_tts.corpus.coverage import CorpusConfig, Sentence, count_sentence, estimate_seconds, load_config
from parrot_tts.text.g2p import g2p
from parrot_tts.text.normalize import normalize

Normalizer = Callable[[str, str], str]  # (văn bản thô, vùng) -> chữ đọc được


@dataclass(frozen=True)
class Problem:
    check: str  # "normalize" | "overlap" | "duplicate" | "duration" | "g2p"
    ids: tuple[str, ...]
    message: str


def normalize_for_compare(text: str) -> str:
    """Dạng so khớp của câu: NFC, chữ thường, dấu câu thành khoảng trắng, gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.split())


def check_overlap(test_set: Sequence[Sentence], script: Sequence[Sentence]) -> list[Problem]:
    """(a) Câu vừa ở bộ test vừa ở kịch bản thu: mỗi cặp (mã test, mã kịch bản) là một vấn đề."""
    in_script: dict[str, list[str]] = {}
    for sentence in script:
        in_script.setdefault(normalize_for_compare(sentence.text), []).append(sentence.id)
    problems = []
    for sentence in test_set:
        key = normalize_for_compare(sentence.text)
        for script_id in in_script.get(key, []):
            problems.append(
                Problem("overlap", (sentence.id, script_id), f"câu test cũng có trong kịch bản thu: {key!r}")
            )
    return problems


def check_duplicates(sentences: Sequence[Sentence]) -> list[Problem]:
    """(d) Câu trùng nhau trong cùng một bộ: mỗi nhóm trùng là một vấn đề."""
    groups: dict[str, list[str]] = {}
    for sentence in sentences:
        groups.setdefault(normalize_for_compare(sentence.text), []).append(sentence.id)
    return [
        Problem("duplicate", tuple(ids), f"{len(ids)} câu trùng nội dung: {key!r}")
        for key, ids in groups.items()
        if len(ids) > 1
    ]


def check_duration(sentences: Sequence[Sentence], dialect: str, config: CorpusConfig | None = None) -> list[Problem]:
    """(b) Thời lượng ước tính (số âm tiết / speech_rate) phải nằm trong [min_seconds, max_seconds].

    Câu không qua G2P thì không đếm được âm tiết, cũng bị báo ở đây.
    """
    config = config or load_config()
    problems = []
    for sentence in sentences:
        try:
            syllables = count_sentence(sentence.text, dialect, config).syllables
        except ValueError as err:
            problems.append(Problem("duration", (sentence.id,), f"không ước tính được thời lượng: {err}"))
            continue
        seconds = estimate_seconds(syllables, config)
        if not config.min_seconds <= seconds <= config.max_seconds:
            problems.append(
                Problem(
                    "duration",
                    (sentence.id,),
                    f"{syllables} âm tiết ≈ {seconds:.2f} giây, ngoài khoảng "
                    f"{config.min_seconds:g}–{config.max_seconds:g} giây",
                )
            )
    return problems


def check_g2p(sentences: Sequence[Sentence], dialect: str) -> list[Problem]:
    """(c) Mọi câu phải qua G2P của ``dialect``; gom hết các câu lỗi."""
    problems = []
    for sentence in sentences:
        try:
            g2p(sentence.text, dialect)
        except ValueError as err:
            problems.append(Problem("g2p", (sentence.id,), str(err)))
    return problems


def normalize_sentences(
    sentences: Sequence[Sentence], dialect: str, normalizer: Normalizer = normalize
) -> tuple[list[Sentence], list[Problem]]:
    """P1 cho từng câu: (các câu đã chuẩn hoá, giữ mã và loại ; vấn đề của câu chuẩn hoá lỗi)."""
    normalized: list[Sentence] = []
    problems: list[Problem] = []
    for sentence in sentences:
        try:
            normalized.append(replace(sentence, text=normalizer(sentence.text, dialect)))
        except ValueError as err:
            problems.append(Problem("normalize", (sentence.id,), str(err)))
    return normalized, problems


def run_checks(
    test_set: Sequence[Sentence],
    script: Sequence[Sentence],
    dialect: str,
    config: CorpusConfig | None = None,
    *,
    normalizer: Normalizer = normalize,
) -> list[Problem]:
    """Chạy mọi kiểm tra trên bộ test và kịch bản thu (văn bản thô) của một vùng.

    Câu lỗi chuẩn hoá chỉ báo ``normalize``; câu lỗi G2P chỉ báo ``g2p``, không báo thêm ``duration``.
    """
    config = config or load_config()
    g2p("", dialect)  # vùng sai là lỗi của người gọi: ném ngay, không báo thành lỗi từng câu
    problems: list[Problem] = []
    raw_sets = (test_set, script)
    norm_sets: list[list[Sentence]] = []
    for sentences in raw_sets:
        normalized, failures = normalize_sentences(sentences, dialect, normalizer)
        norm_sets.append(normalized)
        problems += failures
    problems += _merge(check_overlap(*norm_sets), check_overlap(test_set, script))
    for raw, normalized in zip(raw_sets, norm_sets, strict=True):
        problems += _merge(check_duplicates(normalized), check_duplicates(raw))
        g2p_problems = check_g2p(normalized, dialect)
        failed = {p.ids[0] for p in g2p_problems}
        problems += g2p_problems
        problems += check_duration([s for s in normalized if s.id not in failed], dialect, config)
    return problems


def _merge(normalized: list[Problem], raw: list[Problem]) -> list[Problem]:
    """Gộp vấn đề tìm trên chữ đã chuẩn hoá và chữ thô; bỏ nhóm thô đã nằm trong một nhóm chuẩn hoá."""
    merged = list(normalized)
    for problem in raw:
        if not any(set(problem.ids) <= set(other.ids) for other in normalized):
            merged.append(problem)
    return merged
