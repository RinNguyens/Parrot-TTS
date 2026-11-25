"""Đo độ phủ (âm vị, thanh) và cặp thanh liền kề (plan P3, đặc tả mục 5)."""

import dataclasses
from collections import Counter
from pathlib import Path

import pytest

from vietvoice.corpus.coverage import (
    count_sentence,
    load_config,
    measure,
    pairs_in_tokens,
    render_report,
    tone_pairs_in_tokens,
    valid_pairs,
    valid_tone_pairs,
    valid_tones,
)
from vietvoice.text.g2p import g2p

SENTENCE = "ăn cơm, ăn cá."  # Bắc: ʔ a n T1 _ k ə m T1 , ʔ a n T1 _ k a T5 .


# --- đếm trên một câu đếm tay ---


def test_pairs_counted_by_hand():
    pairs = pairs_in_tokens(g2p(SENTENCE, "bac"))
    assert pairs == Counter(
        {
            ("ʔ", "T1"): 2,
            ("a", "T1"): 2,
            ("n", "T1"): 2,
            ("k", "T1"): 1,
            ("ə", "T1"): 1,
            ("m", "T1"): 1,
            ("k", "T5"): 1,
            ("a", "T5"): 1,
        }
    )


def test_tone_pairs_reset_at_punctuation():
    tokens = g2p(SENTENCE, "bac")
    assert tone_pairs_in_tokens(tokens) == Counter({("T1", "T1"): 1, ("T1", "T5"): 1})


def test_tone_pairs_across_punctuation_when_asked():
    tokens = g2p(SENTENCE, "bac")
    assert tone_pairs_in_tokens(tokens, across_punctuation=True) == Counter({("T1", "T1"): 2, ("T1", "T5"): 1})


def test_tone_pairs_cross_word_boundary():
    # "_" không cắt chuỗi thanh.
    assert tone_pairs_in_tokens(["m", "a", "T1", "_", "m", "a", "T2", "_", "m", "a", "T6"]) == Counter(
        {("T1", "T2"): 1, ("T2", "T6"): 1}
    )


def test_tokens_without_tone_at_end_rejected():
    with pytest.raises(ValueError, match="thanh"):
        pairs_in_tokens(["m", "a"])


def test_count_sentence_syllables():
    counts = count_sentence(SENTENCE, "bac")
    assert counts.syllables == 4
    assert counts.pairs[("ʔ", "T1")] == 2
    assert counts.tone_pairs[("T1", "T5")] == 1


def test_count_sentence_g2p_error_propagates():
    with pytest.raises(ValueError):
        count_sentence("có 2 con", "bac")


# --- tập cặp hợp lệ theo vùng ---


def test_checked_coda_pairs_only_with_t7_t8():
    pairs = valid_pairs("bac")
    for coda in ("k͡p", "ʲk"):
        assert (coda, "T7") in pairs
        assert (coda, "T8") in pairs
        for tone in ("T1", "T2", "T3", "T4", "T5", "T6"):
            assert (coda, tone) not in pairs


def test_nasal_only_coda_never_with_checked_tones():
    pairs = valid_pairs("bac")
    for coda in ("ŋ͡m", "ʲŋ", "j"):
        assert (coda, "T1") in pairs
        assert (coda, "T7") not in pairs
        assert (coda, "T8") not in pairs


def test_onset_and_vowel_take_all_bac_tones():
    pairs = valid_pairs("bac")
    for phoneme in ("k", "a", "ʔ", "w"):
        for tone in range(1, 9):
            assert (phoneme, f"T{tone}") in pairs


def test_nam_has_no_t3():
    assert all(tone != "T3" for _, tone in valid_pairs("nam"))
    assert valid_tones("nam") == ("T1", "T2", "T4", "T5", "T6", "T7", "T8")
    assert ("a", "T4") in valid_pairs("nam")
    assert ("k͡p", "T1") not in valid_pairs("nam")


def test_valid_tone_pairs_per_dialect():
    assert len(valid_tone_pairs("bac")) == 64
    assert len(valid_tone_pairs("nam")) == 49
    assert ("T3", "T1") not in valid_tone_pairs("nam")


def test_valid_pairs_unknown_dialect():
    with pytest.raises(ValueError, match="vùng"):
        valid_pairs("hue")


def test_valid_pairs_cover_real_sentence():
    # Mọi cặp của một câu thật đều nằm trong tập hợp lệ.
    for dialect in ("bac", "nam"):
        text = "học sinh đọc sách ở thư viện, người việt nói tiếng việt."
        assert set(pairs_in_tokens(g2p(text, dialect))) <= valid_pairs(dialect)


# --- báo cáo ---


def test_report_counts_and_missing():
    report = measure([SENTENCE, "ăn cơm"], "bac")
    assert report.sentences == 2
    assert report.syllables == 6
    assert report.pair_counts[("ʔ", "T1")] == 3
    assert report.missing_pairs[("ʔ", "T1")] == 3  # dưới 10
    assert report.missing_pairs[("k͡p", "T8")] == 0
    assert ("k͡p", "T1") not in report.missing_pairs
    assert report.tone_pair_counts[("T1", "T1")] == 2
    assert ("T1", "T1") not in report.missing_tone_pairs
    assert ("T1", "T5") not in report.missing_tone_pairs  # đã có 1 lần, ngưỡng 1
    assert ("T3", "T3") in report.missing_tone_pairs
    assert not report.complete
    assert not report.unexpected_pairs


def test_report_respects_config_minimum():
    config = dataclasses.replace(load_config(), min_pair_count=2)
    report = measure([SENTENCE], "bac", config=config)
    assert ("ʔ", "T1") not in report.missing_pairs
    assert report.missing_pairs[("k", "T1")] == 1


def test_report_complete_when_every_target_met():
    config = dataclasses.replace(load_config(), min_pair_count=0, min_tone_pair_count=0)
    assert measure([], "nam", config=config).complete


def test_render_report_bac():
    text = render_report(measure([SENTENCE], "bac"))
    assert "Bắc" in text
    assert "64" in text
    assert str(len(valid_pairs("bac"))) in text


def test_render_report_nam_mentions_missing_t3():
    text = render_report(measure([SENTENCE], "nam"))
    assert "Nam" in text
    assert "T3" in text
    assert "49" in text


# --- cấu hình ---


def test_default_config_values():
    config = load_config()
    assert config.min_pair_count == 10
    assert config.speech_rate == 5.6
    assert config.coverage_category in config.quotas
    assert abs(sum(config.quotas.values()) - 1) < 1e-9


def test_config_quotas_must_sum_to_one(tmp_path: Path):
    raw = (Path(__file__).parents[1] / "src/vietvoice/corpus/config.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "config.yaml"
    bad.write_text(raw.replace("phu_am_vi: 0.30", "phu_am_vi: 0.50"), encoding="utf-8")
    with pytest.raises(ValueError, match="tổng"):
        load_config(bad)


def test_config_missing_key(tmp_path: Path):
    bad = tmp_path / "config.yaml"
    bad.write_text("coverage: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config.yaml"):
        load_config(bad)
