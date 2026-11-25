"""Chỉ số tự động của P6 (đặc tả mục 10), test bằng giá trị biết trước.

Đầu vào là kết quả đã tính sẵn (bản ASR, nhãn thanh, mảng F0, embedding); không cần mô hình.
"""

import unicodedata
from pathlib import Path

import numpy as np
import pytest

from parrot_tts.eval.metrics import (
    CheckpointScores,
    align_tones,
    cer,
    creak_rate_gap,
    dtw,
    f0_rmse_semitones,
    hz_to_semitones,
    load_thresholds,
    normalize_text,
    speaker_similarity,
    speech_rate,
    speech_rate_deviation,
    tone_confusion_matrix,
    tone_error_rate,
    wer,
)

# ---------------------------------------------------------------- CER, WER


def test_normalize_text_nfc_lowercase_whitespace():
    text = unicodedata.normalize("NFD", "  Xin   CHÀO\tbạn\n")
    assert normalize_text(text) == unicodedata.normalize("NFC", "xin chào bạn")


def test_cer_identical_is_zero():
    assert cer("mẹ đi chợ!", "mẹ đi chợ!") == 0.0


def test_cer_one_substitution_in_ten_chars():
    assert cer("abcdefghij", "abcdefghik") == pytest.approx(0.1)


def test_cer_tone_error_is_one_character():
    # "mẹ đi chợ!" có 10 ký tự (tính cả dấu cách); sai một thanh = thay một ký tự.
    assert cer("mẹ đi chợ!", "mẹ đi chờ!") == pytest.approx(0.1)


def test_cer_insertion_and_deletion():
    assert cer("abcdefghij", "abcdefghijk") == pytest.approx(0.1)
    assert cer("abcdefghij", "abcdefghi") == pytest.approx(0.1)
    assert cer("abcdefghij", "bcdefghij") == pytest.approx(0.1)


def test_cer_can_exceed_one():
    assert cer("ab", "abxyz") == pytest.approx(1.5)


def test_cer_empty_hypothesis_is_one():
    assert cer("xin chào", "") == pytest.approx(1.0)


def test_cer_composed_equals_decomposed():
    ref = "người Việt nói tiếng Việt"
    hyp = "người việt nói tiến việt"
    nfc = cer(unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp))
    nfd = cer(unicodedata.normalize("NFD", ref), unicodedata.normalize("NFD", hyp))
    assert nfc == nfd
    assert nfc == pytest.approx(1 / 25)


def test_cer_ignores_case_and_extra_whitespace():
    assert cer("Xin  chào\n", "xin chào") == 0.0


def test_cer_spaces_are_characters():
    # Theo quy ước CER của jiwer: dấu cách là một ký tự. "xin chào" có 8 ký tự.
    assert cer("xin chào", "xinchào") == pytest.approx(1 / 8)


@pytest.mark.parametrize("ref", ["", "   ", "\n\t"])
def test_cer_empty_reference_raises(ref):
    with pytest.raises(ValueError, match="rỗng"):
        cer(ref, "abc")


def test_wer_values():
    assert wer("tôi đi học", "tôi đi học") == 0.0
    assert wer("tôi đi học", "tôi đi chợ") == pytest.approx(1 / 3)
    assert wer("tôi đi học", "tôi học") == pytest.approx(1 / 3)
    assert wer("tôi đi học", "tôi đi học về") == pytest.approx(1 / 3)
    assert wer("tôi đi học", "") == pytest.approx(1.0)


def test_wer_composed_equals_decomposed():
    ref = "Hà Nội mùa thu"
    hyp = "hà nội mùa thu"
    assert wer(unicodedata.normalize("NFD", ref), hyp) == 0.0


def test_wer_empty_reference_raises():
    with pytest.raises(ValueError, match="rỗng"):
        wer("  ", "abc")


# ---------------------------------------------------------------- TER, ma trận nhầm lẫn


def test_ter_identical_is_zero():
    assert tone_error_rate([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]) == 0.0


def test_ter_deleted_syllable():
    assert tone_error_rate([1, 2, 3, 4], [1, 3, 4]) == pytest.approx(0.25)


def test_ter_inserted_syllable_and_substitution():
    assert tone_error_rate([1, 2, 3, 4], [1, 2, 5, 3, 4]) == pytest.approx(0.25)
    assert tone_error_rate([1, 2, 3, 4], [1, 2, 4, 4]) == pytest.approx(0.25)


def test_ter_accepts_numpy_ints():
    assert tone_error_rate(np.array([3, 6]), np.array([3, 8])) == pytest.approx(0.5)


def test_ter_empty_reference_raises():
    with pytest.raises(ValueError, match="rỗng"):
        tone_error_rate([], [1])


@pytest.mark.parametrize("bad", [[0], [9], [1.5], ["1"], [True]])
def test_ter_rejects_values_outside_1_to_8(bad):
    with pytest.raises(ValueError):
        tone_error_rate(bad, [1])
    with pytest.raises(ValueError):
        tone_error_rate([1], bad)


def test_align_tones_marks_deletion_and_insertion():
    assert align_tones([1, 2, 3, 4], [1, 3, 4]) == [(1, 1), (2, None), (3, 3), (4, 4)]
    assert align_tones([1, 2], [1, 5, 2]) == [(1, 1), (None, 5), (2, 2)]
    assert align_tones([6], [8]) == [(6, 8)]


def test_confusion_matrix_hand_made_case():
    pairs = [(1, 1), (1, 1), (3, 4), (6, 8), (8, 8), (2, None), (None, 5)]
    matrix = tone_confusion_matrix(pairs)
    assert matrix.shape == (8, 8)
    assert np.issubdtype(matrix.dtype, np.integer)
    assert matrix[0, 0] == 2  # ngang -> ngang
    assert matrix[2, 3] == 1  # ngã bị đọc thành hỏi
    assert matrix[5, 7] == 1  # nặng -> nặng tắc
    assert matrix[7, 7] == 1
    assert matrix.sum() == 5  # cặp có None (thêm, bớt) không vào ma trận


def test_confusion_matrix_from_alignment():
    matrix = tone_confusion_matrix(align_tones([3, 3, 4], [4, 3]))
    assert matrix.sum() == 2
    assert matrix[2, 2] + matrix[2, 3] == 2


@pytest.mark.parametrize("pair", [(0, 1), (1, 9), (None, None), (2.0, 2)])
def test_confusion_matrix_rejects_bad_pairs(pair):
    with pytest.raises(ValueError):
        tone_confusion_matrix([pair])


# ---------------------------------------------------------------- F0, DTW


def test_hz_to_semitones():
    assert hz_to_semitones(np.array([100.0, 200.0, 50.0]), 100.0) == pytest.approx([0.0, 12.0, -12.0])


def test_hz_to_semitones_bad_baseline_raises():
    with pytest.raises(ValueError, match="baseline"):
        hz_to_semitones(np.array([100.0]), 0.0)


def _contour(n: int = 40) -> np.ndarray:
    # Đường thanh hỏi giả: xuống rồi lên, quanh 200 Hz.
    t = np.linspace(0.0, 1.0, n)
    return 200.0 * 2 ** ((-3 * np.sin(np.pi * t) + 2 * t) / 12)


def test_f0_rmse_identical_is_zero():
    f0 = _contour()
    assert f0_rmse_semitones(f0, f0.copy(), 200.0) == pytest.approx(0.0, abs=1e-12)


def test_f0_rmse_constant_one_semitone_shift():
    f0 = np.full(25, 180.0)
    assert f0_rmse_semitones(f0, f0 * 2 ** (1 / 12), 200.0) == pytest.approx(1.0)


def test_f0_rmse_dtw_absorbs_part_of_shift_on_moving_contour():
    # DTW căn trên chính F0 nên kéo đường lệch về gần nhau: lệch đều 1 semitone
    # trên đường cong đo ra < 1. Giữ test này để ai đổi cách căn đều thấy.
    f0 = _contour()
    rmse = f0_rmse_semitones(f0, f0 * 2 ** (1 / 12), 200.0)
    assert 0.0 < rmse < 1.0


def test_f0_rmse_constant_offset_flat_curves():
    ref = np.full(20, 100.0)
    hyp = np.full(30, 200.0)
    assert f0_rmse_semitones(ref, hyp, 100.0) == pytest.approx(12.0)


def test_f0_rmse_time_stretched_copy_is_zero():
    f0 = _contour()
    stretched = np.repeat(f0, 2)
    assert f0_rmse_semitones(f0, stretched, 200.0) == pytest.approx(0.0, abs=1e-12)
    assert f0_rmse_semitones(stretched, f0, 200.0) == pytest.approx(0.0, abs=1e-12)


def test_f0_rmse_ignores_unvoiced_frames():
    f0 = _contour()
    hyp = np.concatenate([[0.0, 0.0], f0[:20], [0.0, -1.0, np.nan], f0[20:], [0.0]])
    assert f0_rmse_semitones(f0, hyp, 200.0) == pytest.approx(0.0, abs=1e-12)


def test_f0_rmse_all_unvoiced_raises():
    with pytest.raises(ValueError, match="hữu thanh"):
        f0_rmse_semitones(np.zeros(10), _contour(), 200.0)
    with pytest.raises(ValueError, match="hữu thanh"):
        f0_rmse_semitones(_contour(), np.array([0.0, -5.0, np.nan]), 200.0)


def test_f0_rmse_rejects_2d_input():
    with pytest.raises(ValueError, match="1 chiều"):
        f0_rmse_semitones(np.ones((2, 3)) * 100, _contour(), 200.0)


def test_f0_rmse_rejects_bad_baseline():
    with pytest.raises(ValueError, match="baseline"):
        f0_rmse_semitones(_contour(), _contour(), -1.0)


def test_dtw_known_small_case():
    cost, path = dtw(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0]))
    # Đường tốt nhất: (0,0) (1,1) (2,1) hoặc (0,0) (1,0) (2,1); cả hai tổng chi phí 1.
    assert cost == pytest.approx(1.0)
    assert path[0] == (0, 0)
    assert path[-1] == (2, 1)


def test_dtw_path_is_monotonic_and_continuous():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=15), rng.normal(size=9)
    _, path = dtw(a, b)
    assert path[0] == (0, 0) and path[-1] == (14, 8)
    for (i0, j0), (i1, j1) in zip(path, path[1:], strict=False):
        assert (i1 - i0, j1 - j0) in {(1, 0), (0, 1), (1, 1)}


def test_dtw_empty_raises():
    with pytest.raises(ValueError):
        dtw(np.array([]), np.array([1.0]))


# ---------------------------------------------------------------- creak


def test_creak_rate_gap_percentage_points():
    real = [True, False, False, False]  # 25%
    synth = [True, True, False, False]  # 50%
    assert creak_rate_gap(real, synth) == pytest.approx(25.0)
    assert creak_rate_gap(synth, real) == pytest.approx(-25.0)


def test_creak_rate_gap_different_lengths():
    assert creak_rate_gap([True, False], [False, False, False, False, True]) == pytest.approx(-30.0)


def test_creak_rate_gap_accepts_numpy_bool_and_0_1():
    assert creak_rate_gap(np.array([1, 0]), np.array([True, True])) == pytest.approx(50.0)


@pytest.mark.parametrize(("ref", "hyp"), [([], [True]), ([True], []), ([2], [True]), ([0.5], [True])])
def test_creak_rate_gap_bad_input_raises(ref, hyp):
    with pytest.raises(ValueError):
        creak_rate_gap(ref, hyp)


# ---------------------------------------------------------------- speaker similarity


def test_speaker_similarity_values():
    assert speaker_similarity(np.array([1.0, 0.0]), np.array([0.0, 3.0])) == pytest.approx(0.0)
    assert speaker_similarity(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert speaker_similarity(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.0)
    assert speaker_similarity(np.array([1.0, 2.0]), np.array([-1.0, -2.0])) == pytest.approx(-1.0)


def test_speaker_similarity_zero_vector_raises():
    with pytest.raises(ValueError, match="0"):
        speaker_similarity(np.zeros(3), np.ones(3))


@pytest.mark.parametrize(
    ("a", "b"),
    [(np.ones(3), np.ones(4)), (np.ones((2, 2)), np.ones((2, 2))), (np.array([np.nan, 1.0]), np.ones(2))],
)
def test_speaker_similarity_bad_shape_raises(a, b):
    with pytest.raises(ValueError):
        speaker_similarity(a, b)


# ---------------------------------------------------------------- tốc độ nói


def test_speech_rate():
    assert speech_rate(10, 2.0) == pytest.approx(5.0)
    assert speech_rate(0, 1.0) == 0.0


@pytest.mark.parametrize(("n", "d"), [(10, 0.0), (10, -1.0), (-1, 1.0), (10, float("nan"))])
def test_speech_rate_bad_input_raises(n, d):
    with pytest.raises(ValueError):
        speech_rate(n, d)


def test_speech_rate_deviation_signed_fraction():
    assert speech_rate_deviation(5.5, 5.0) == pytest.approx(0.1)
    assert speech_rate_deviation(4.5, 5.0) == pytest.approx(-0.1)
    assert speech_rate_deviation(5.0, 5.0) == 0.0


def test_speech_rate_deviation_zero_real_raises():
    with pytest.raises(ValueError):
        speech_rate_deviation(5.0, 0.0)


# ---------------------------------------------------------------- tổng hợp checkpoint

GOOD = {
    "cer": 0.01,
    "ter": 0.02,
    "f0_rmse_st": 1.5,
    "creak_gap_pts": -4.0,
    "speaker_similarity": 0.8,
    "speech_rate_dev": 0.05,
}


def test_default_thresholds_match_spec():
    t = load_thresholds()
    assert t["cer"].limit == pytest.approx(0.03) and t["cer"].op == "max"
    assert t["ter"].limit == pytest.approx(0.03) and t["ter"].op == "max"
    assert t["f0_rmse_st"].limit == pytest.approx(2.0) and t["f0_rmse_st"].op == "max"
    assert t["creak_gap_pts"].limit == pytest.approx(10.0) and t["creak_gap_pts"].op == "abs_max"
    assert t["speaker_similarity"].limit == pytest.approx(0.75) and t["speaker_similarity"].op == "min"
    assert t["speech_rate_dev"].limit == pytest.approx(0.10) and t["speech_rate_dev"].op == "abs_max"


def test_checkpoint_all_good_passes():
    scores = CheckpointScores(checkpoint="step_5000", **GOOD)
    assert scores.passed()
    assert all(scores.verdicts().values())


def test_checkpoint_boundary_values_pass():
    scores = CheckpointScores(
        checkpoint="b",
        cer=0.03,
        ter=0.03,
        f0_rmse_st=2.0,
        creak_gap_pts=-10.0,
        speaker_similarity=0.75,
        speech_rate_dev=0.10,
    )
    assert scores.passed()


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("cer", 0.031),
        ("ter", 0.05),
        ("f0_rmse_st", 2.1),
        ("creak_gap_pts", -12.0),
        ("creak_gap_pts", 12.0),
        ("speaker_similarity", 0.74),
        ("speech_rate_dev", -0.11),
    ],
)
def test_checkpoint_one_bad_metric_fails(metric, value):
    scores = CheckpointScores(checkpoint="x", **{**GOOD, metric: value})
    verdicts = scores.verdicts()
    assert verdicts[metric] is False
    assert sum(v is True for v in verdicts.values()) == len(GOOD) - 1
    assert not scores.passed()


def test_checkpoint_missing_metric_is_none_and_not_passed():
    partial = {k: v for k, v in GOOD.items() if k != "speaker_similarity"}
    scores = CheckpointScores(checkpoint="x", **partial)
    assert scores.verdicts()["speaker_similarity"] is None
    assert scores.missing() == ["speaker_similarity"]
    assert not scores.passed()


def test_checkpoint_nan_score_raises():
    scores = CheckpointScores(checkpoint="x", **{**GOOD, "cer": float("nan")})
    with pytest.raises(ValueError, match="cer"):
        scores.verdicts()


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "metrics.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_custom_thresholds_file(tmp_path):
    lines = [f"{k}: {{max: 1.0}}" for k in GOOD if k != "speaker_similarity"]
    lines.append("speaker_similarity: {min: 0.9}")
    thresholds = load_thresholds(_write(tmp_path, "\n".join(lines)))
    scores = CheckpointScores(checkpoint="x", **GOOD)
    assert scores.verdicts(thresholds)["speaker_similarity"] is False
    assert scores.verdicts(thresholds)["creak_gap_pts"] is True  # max, không lấy trị tuyệt đối


@pytest.mark.parametrize(
    "text",
    [
        "cer: {max: 0.03}",  # thiếu chỉ số khác
        "\n".join([f"{k}: {{max: 1}}" for k in GOOD] + ["utmos: {min: 3}"]),  # chỉ số lạ
        "\n".join([f"{k}: {{max: 1, min: 0}}" for k in GOOD]),  # hai phép so sánh
        "\n".join([f"{k}: {{below: 1}}" for k in GOOD]),  # phép lạ
        "\n".join([f"{k}: {{max: abc}}" for k in GOOD]),  # không phải số
        "- 1\n- 2",  # không phải bảng
    ],
)
def test_bad_thresholds_file_raises(tmp_path, text):
    with pytest.raises(ValueError):
        load_thresholds(_write(tmp_path, text))
