"""Chọn câu tham lam theo độ phủ và chia quota theo loại câu (đặc tả mục 5)."""

import dataclasses

import pytest

from vietvoice.corpus.coverage import Sentence, count_sentence, load_config
from vietvoice.corpus.select import (
    StopReason,
    Unit,
    category_budget,
    greedy_pick,
    select_coverage_share,
    select_sentences,
)


def unit(uid: str, counts: dict[str, int], syllables: int, seconds: float | None = None) -> Unit:
    return Unit(id=uid, counts=counts, syllables=syllables, seconds=float(syllables) if seconds is None else seconds)


POOL = [
    unit("U1", {"a": 1}, 2),
    unit("U2", {"a": 2, "b": 1}, 5),
    unit("U3", {"b": 1, "c": 1}, 3),
    unit("U4", {"c": 1}, 1),
]
NEED = {"a": 2, "b": 1, "c": 1}


# --- lõi tham lam trên khoá trừu tượng, thứ tự biết trước ---


def test_greedy_known_order():
    result = greedy_pick(POOL, NEED)
    # U2 lợi 3 trước; sau đó U3 và U4 cùng lợi 1, U4 ngắn hơn.
    assert result.ids == ("U2", "U4")
    assert result.stop_reason is StopReason.TARGETS_MET
    assert result.remaining == {}
    assert result.seconds == 6.0


def test_gain_capped_by_remaining_need():
    pool = [unit("X", {"a": 5}, 1), unit("Y", {"a": 1, "b": 1}, 9)]
    # X chỉ được tính lợi 1 vì cần thêm 1 lần "a"; Y lợi 2.
    assert greedy_pick(pool, {"a": 1, "b": 1}).ids == ("Y",)


def test_tie_same_length_keeps_pool_order():
    pool = [unit("B", {"x": 1}, 3), unit("A", {"y": 1}, 3)]
    assert greedy_pick(pool, {"x": 1, "y": 1}).ids == ("B", "A")


def test_tie_prefers_shorter():
    pool = [unit("long", {"x": 1}, 9), unit("short", {"y": 1}, 2)]
    assert greedy_pick(pool, {"x": 1, "y": 1}, max_sentences=1).ids == ("short",)


def test_stop_at_max_sentences():
    result = greedy_pick(POOL, NEED, max_sentences=1)
    assert result.ids == ("U2",)
    assert result.stop_reason is StopReason.MAX_SENTENCES
    assert result.remaining == {"c": 1}


def test_stop_at_max_seconds_skips_what_does_not_fit():
    # U2 (5 s) không vừa ngân sách 4 s; U3 (3 s) vừa; U1 (2 s) còn có ích nhưng không vừa 1 s còn lại.
    result = greedy_pick(POOL, NEED, max_seconds=4.0)
    assert result.ids == ("U3",)
    assert result.stop_reason is StopReason.MAX_SECONDS
    assert result.remaining == {"a": 2}


def test_seconds_budget_fits_short_sentence_after_skip():
    result = greedy_pick(POOL, {"a": 2, "b": 1, "c": 1}, max_seconds=5.0)
    # U2 (5 s) vừa đúng ngân sách và lợi lớn nhất.
    assert result.ids == ("U2",)
    assert result.stop_reason is StopReason.MAX_SECONDS


def test_stop_when_nothing_adds_gain():
    result = greedy_pick(POOL, {"z": 1})
    assert result.ids == ()
    assert result.stop_reason is StopReason.NO_GAIN


def test_stop_when_pool_exhausted():
    result = greedy_pick([unit("U1", {"a": 1}, 2)], {"a": 3})
    assert result.ids == ("U1",)
    assert result.stop_reason is StopReason.POOL_EXHAUSTED
    assert result.remaining == {"a": 2}


def test_targets_already_met():
    result = greedy_pick(POOL, {"a": 0})
    assert result.ids == ()
    assert result.stop_reason is StopReason.TARGETS_MET


def test_bad_budget_rejected():
    with pytest.raises(ValueError):
        greedy_pick(POOL, NEED, max_sentences=-1)
    with pytest.raises(ValueError):
        greedy_pick(POOL, NEED, max_seconds=-0.5)


def test_duplicate_unit_ids_rejected():
    with pytest.raises(ValueError, match="U1"):
        greedy_pick([unit("U1", {"a": 1}, 1), unit("U1", {"b": 1}, 1)], NEED)


def test_greedy_matches_brute_force_order():
    # So với tham lam tính lại toàn bộ mỗi bước (không lười) trên một pool lớn hơn.
    pool = [unit(f"S{i}", {k: (i * 7 + j) % 3 for j, k in enumerate("abcdefg")}, 1 + (i * 5) % 11) for i in range(40)]
    need = dict.fromkeys("abcdefg", 6)
    expected: list[str] = []
    remaining = dict(need)
    left = list(pool)
    while any(remaining.values()) and left:
        scored = [
            (-sum(min(c, remaining.get(k, 0)) for k, c in u.counts.items()), u.syllables, i) for i, u in enumerate(left)
        ]
        best = min(scored)
        if best[0] == 0:
            break
        chosen = left.pop(best[2])
        expected.append(chosen.id)
        for k, c in chosen.counts.items():
            if k in remaining:
                remaining[k] = max(0, remaining[k] - c)
    assert list(greedy_pick(pool, need).ids) == expected


# --- chọn trên câu thật ---

SENTENCES = [
    Sentence("s1", "ăn cơm chưa", "phu_am_vi"),
    Sentence("s2", "giá vàng hôm nay tăng mạnh", "phu_am_vi"),
    Sentence("s3", "học sinh đọc sách ở thư viện", "phu_am_vi"),
    Sentence("s4", "quả xoài này ngọt lắm", "phu_am_vi"),
    Sentence("s5", "người việt nói tiếng việt", "tin_tuc"),
    Sentence("s6", "ăn cơm chưa", "phu_am_vi"),
]


def test_select_sentences_deterministic():
    first = select_sentences(SENTENCES, "bac")
    second = select_sentences(list(SENTENCES), "bac")
    assert first == second
    assert len(set(first.ids)) == len(first.ids)
    # Ngưỡng 10 lần chưa đạt với pool nhỏ: mọi câu có lợi đều được chọn, trừ s6 (trùng s1).
    assert set(first.ids) == {s.id for s in SENTENCES} - {"s6"}
    assert first.dropped_duplicates == (("s6", "s1"),)
    assert first.stop_reason is StopReason.POOL_EXHAUSTED


def test_duplicates_dropped_after_normalizing():
    pool = [
        Sentence("raw", "Giảm giá 50% hôm nay!"),
        Sentence("spelled", "giảm giá năm mươi phần trăm hôm nay"),
        Sentence("other", "ăn cơm chưa"),
    ]
    result = select_sentences(pool, "bac")
    assert set(result.ids) == {"raw", "other"}
    assert result.dropped_duplicates == (("spelled", "raw"),)


def test_raw_digits_normalized_before_g2p():
    result = select_sentences([Sentence("d", "có 2 con mèo")], "bac")
    assert result.ids == ("d",)


def test_normalize_errors_collected():
    def fake_normalize(text: str, dialect: str) -> str:
        if "hỏng" in text:
            raise ValueError("chuẩn hoá lỗi")
        return text

    pool = [Sentence("ok", "ăn cơm"), Sentence("n1", "câu hỏng"), Sentence("n2", "hỏng nữa"), Sentence("g", "zzz")]
    with pytest.raises(ValueError) as err:
        select_sentences(pool, "bac", normalizer=fake_normalize)
    assert all(i in str(err.value) for i in ("n1", "n2", "g"))
    assert "ok:" not in str(err.value)


def test_select_sentences_first_pick_has_max_gain():
    config = load_config()
    result = select_sentences(SENTENCES, "nam", max_sentences=1)
    gains = {}
    for s in SENTENCES:
        c = count_sentence(s.text, "nam", config)
        gains[s.id] = sum(c.pairs.values()) + sum(c.tone_pairs.values())
    best = max(gains.values())
    assert gains[result.ids[0]] == best


def test_already_counted_texts_reduce_need():
    config = dataclasses.replace(load_config(), min_pair_count=1, min_tone_pair_count=1)
    pool = [Sentence("a", "ăn cơm chưa"), Sentence("b", "quả xoài này ngọt lắm")]
    result = select_sentences(pool, "bac", already=["ăn cơm chưa"], config=config, max_sentences=5)
    assert "a" not in result.ids
    assert "b" in result.ids


def test_select_sentences_collects_all_g2p_errors():
    pool = [Sentence("ok", "ăn cơm"), Sentence("bad1", "blah blah"), Sentence("bad2", "xyzw qrst")]
    with pytest.raises(ValueError) as err:
        select_sentences(pool, "bac")
    assert "bad1" in str(err.value) and "bad2" in str(err.value)


def test_select_sentences_seconds_from_speech_rate():
    config = dataclasses.replace(load_config(), min_pair_count=1)
    result = select_sentences([Sentence("a", "ăn cơm chưa")], "bac", config=config)
    assert result.seconds == pytest.approx(3 / 5.6)


# --- quota theo loại câu ---


def test_category_budget_spec_table():
    assert category_budget(100) == {
        "phu_am_vi": 30,
        "quang_cao": 25,
        "tin_tuc": 20,
        "ke_chuyen": 15,
        "hoi_cam_than": 5,
        "tu_vung_mien": 5,
    }


@pytest.mark.parametrize("total", [0, 1, 7, 13, 999, 10001])
def test_category_budget_sums_to_total(total: int):
    budget = category_budget(total)
    assert sum(budget.values()) == total
    assert budget == category_budget(total)
    for name, share in load_config().quotas.items():
        assert abs(budget[name] - share * total) < 1


def test_category_budget_negative_rejected():
    with pytest.raises(ValueError):
        category_budget(-1)


def test_coverage_share_uses_30_percent_and_only_coverage_category():
    config = dataclasses.replace(load_config(), min_pair_count=50)
    pool = [Sentence(f"c{i}", s.text, "phu_am_vi") for i, s in enumerate(SENTENCES[:4])]
    pool += [Sentence("news", "người việt nói tiếng việt", "tin_tuc")]
    result = select_coverage_share(pool, "bac", total_sentences=10, config=config)
    assert len(result.ids) == 3  # 30% của 10
    assert "news" not in result.ids
    assert result.stop_reason is StopReason.MAX_SENTENCES


def test_coverage_share_seconds_budget():
    config = dataclasses.replace(load_config(), min_pair_count=50)
    pool = [Sentence(f"c{i}", s.text, "phu_am_vi") for i, s in enumerate(SENTENCES[:4])]
    result = select_coverage_share(pool, "bac", total_sentences=100, total_seconds=5.0, config=config)
    assert result.seconds <= 5.0 * 0.30 + 1e-9
