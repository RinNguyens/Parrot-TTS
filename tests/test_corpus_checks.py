"""Kiểm tra bộ câu test và kịch bản thu (plan P3)."""

import dataclasses

import pytest

from vietvoice.corpus.checks import (
    Problem,
    check_duplicates,
    check_duration,
    check_g2p,
    check_overlap,
    normalize_for_compare,
    run_checks,
)
from vietvoice.corpus.coverage import Sentence, load_config


def test_normalize_for_compare():
    assert normalize_for_compare("  Ăn   CƠM,\tchưa?! ") == "ăn cơm chưa"
    # NFD và NFC so khớp như nhau.
    assert normalize_for_compare("cơm") == normalize_for_compare("cơm")


# --- (a) câu vừa ở bộ test vừa ở kịch bản ---


def test_overlap_caught_after_normalizing():
    test_set = [Sentence("t1", "Ăn cơm chưa?"), Sentence("t2", "trời hôm nay đẹp quá")]
    script = [Sentence("r1", "giá vàng tăng"), Sentence("r2", "ăn  cơm, chưa")]
    problems = check_overlap(test_set, script)
    assert len(problems) == 1
    assert problems[0].check == "overlap"
    assert problems[0].ids == ("t1", "r2")


def test_overlap_reports_every_pair():
    test_set = [Sentence("t1", "ăn cơm"), Sentence("t2", "ăn cơm.")]
    script = [Sentence("r1", "Ăn cơm")]
    assert [p.ids for p in check_overlap(test_set, script)] == [("t1", "r1"), ("t2", "r1")]


def test_no_overlap():
    assert check_overlap([Sentence("t1", "ăn cơm")], [Sentence("r1", "ăn cá")]) == []


# --- (d) câu trùng trong một bộ ---


def test_duplicates_within_set():
    sentences = [
        Sentence("a", "Ăn cơm chưa"),
        Sentence("b", "giá vàng tăng"),
        Sentence("c", "ăn cơm chưa?"),
        Sentence("d", "ĂN CƠM CHƯA"),
    ]
    problems = check_duplicates(sentences)
    assert len(problems) == 1
    assert problems[0] == Problem(check="duplicate", ids=("a", "c", "d"), message=problems[0].message)
    assert "ăn cơm chưa" in problems[0].message


def test_no_duplicates():
    assert check_duplicates([Sentence("a", "ăn cơm"), Sentence("b", "ăn cá")]) == []


# --- (b) độ dài ước tính ---


def test_duration_flags_too_short_and_too_long():
    long_text = " ".join(["ba"] * 90)  # 90 / 5,6 ≈ 16,1 giây
    sentences = [
        Sentence("short", "ừ"),  # 1 / 5,6 ≈ 0,18 giây
        Sentence("ok", "giá vàng hôm nay tăng mạnh lắm"),  # 7 âm tiết ≈ 1,25 giây
        Sentence("long", long_text),
    ]
    problems = check_duration(sentences, "bac")
    assert [p.ids for p in problems] == [("short",), ("long",)]
    assert all(p.check == "duration" for p in problems)


def test_duration_uses_config_speech_rate():
    config = dataclasses.replace(load_config(), speech_rate=2.0)
    assert check_duration([Sentence("x", "ăn cơm chưa")], "bac", config) == []  # 1,5 giây


def test_duration_reports_unparsable_sentence():
    problems = check_duration([Sentence("bad", "có 2 con")], "bac")
    assert [p.ids for p in problems] == [("bad",)]


# --- (c) mọi câu qua G2P ---


def test_g2p_collects_every_failure():
    sentences = [
        Sentence("ok", "ăn cơm chưa"),
        Sentence("digit", "có 2 con mèo"),
        Sentence("ok2", "giá vàng"),
        Sentence("junk", "xyzw qrst"),
    ]
    problems = check_g2p(sentences, "nam")
    assert [p.ids for p in problems] == [("digit",), ("junk",)]
    assert all(p.check == "g2p" and p.message for p in problems)


def test_g2p_all_pass():
    assert check_g2p([Sentence("ok", "ăn cơm chưa")], "bac") == []


# --- chạy tất cả ---


def test_run_checks_finds_each_planted_problem():
    test_set = [Sentence("t1", "giá vàng hôm nay tăng mạnh"), Sentence("t2", "Giá vàng hôm nay tăng mạnh.")]
    script = [Sentence("r1", "giá vàng hôm nay tăng mạnh"), Sentence("r2", "xyzw qrst")]
    problems = run_checks(test_set, script, "bac")
    kinds = {(p.check, p.ids) for p in problems}
    assert ("overlap", ("t1", "r1")) in kinds
    assert ("duplicate", ("t1", "t2")) in kinds
    assert ("g2p", ("r2",)) in kinds


def test_run_checks_normalizes_before_g2p():
    # P1 trước P2: chữ số đọc được sau chuẩn hoá, không phải lỗi G2P.
    problems = run_checks([Sentence("t1", "Giảm giá 50% hôm nay cho mọi khách")], [], "bac")
    assert problems == []


def test_run_checks_reports_g2p_failure_once():
    problems = run_checks([], [Sentence("r2", "xyzw qrst")], "bac")
    assert [(p.check, p.ids) for p in problems] == [("g2p", ("r2",))]


def test_run_checks_overlap_on_normalized_text():
    test_set = [Sentence("t1", "Giảm giá 50% hôm nay")]
    script = [Sentence("r1", "giảm giá năm mươi phần trăm hôm nay")]
    problems = run_checks(test_set, script, "bac")
    assert [(p.check, p.ids) for p in problems] == [("overlap", ("t1", "r1"))]


def test_run_checks_duplicates_on_normalized_text_reported_once():
    sentences = [
        Sentence("a", "Giảm giá 50% hôm nay"),
        Sentence("b", "giảm giá năm mươi phần trăm hôm nay"),
        Sentence("c", "giảm giá 50% hôm nay."),
    ]
    problems = run_checks(sentences, [], "bac")
    # Nhóm thô {a, c} nằm trong nhóm đã chuẩn hoá {a, b, c}: chỉ báo một lần.
    assert [(p.check, p.ids) for p in problems] == [("duplicate", ("a", "b", "c"))]


def test_run_checks_normalize_failure_is_own_problem():
    def fake_normalize(text: str, dialect: str) -> str:
        if "hỏng" in text:
            raise ValueError("chuẩn hoá lỗi")
        return text

    sentences = [Sentence("ok", "giá vàng hôm nay tăng mạnh"), Sentence("bad", "câu này hỏng rồi")]
    problems = run_checks(sentences, [], "bac", normalizer=fake_normalize)
    assert [(p.check, p.ids) for p in problems] == [("normalize", ("bad",))]


def test_run_checks_unknown_dialect_raises():
    with pytest.raises(ValueError, match="vùng"):
        run_checks([Sentence("a", "ăn cơm chưa")], [], "hue")
