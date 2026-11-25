"""Bước 3 pipeline: cắt câu ở khoảng lặng >= 300 ms, giữ 0,1–0,2 s lặng hai đầu.

API: ``parrot_tts.data.vad``: Protocol ``VoiceActivityDetector``, ``EnergyVAD``,
``segment_utterances``, ``cut_file``.
"""

import numpy as np
import pytest
import soundfile as sf
from fixtures_audio import SR, noise, sha256, speech_like, write_wav

from parrot_tts.data.vad import (
    EnergyVAD,
    SpeechRegion,
    UtteranceCut,
    VoiceActivityDetector,
    cut_file,
    load_vad_config,
    main,
    segment_utterances,
)

CFG = load_vad_config()
FRAME = round(CFG.energy.frame_ms * SR / 1000)
PAD = round(CFG.pad_s * SR)


def _s(seconds: float) -> int:
    return round(seconds * SR)


def test_config_values_from_spec():
    assert CFG.min_silence_s == 0.3
    assert 0.1 <= CFG.min_pad_s <= CFG.pad_s <= 0.2


def test_energy_vad_satisfies_protocol():
    vad: VoiceActivityDetector = EnergyVAD(CFG.energy)
    assert isinstance(vad, VoiceActivityDetector)


def test_energy_vad_finds_planted_speech_within_one_frame():
    planted = [(0.5, 1.7), (2.3, 3.1)]
    data = speech_like(planted, total_s=3.6)
    regions = EnergyVAD(CFG.energy).detect(data, SR)
    assert len(regions) == 2
    for region, (start_s, end_s) in zip(regions, planted, strict=True):
        assert abs(region.start - _s(start_s)) <= FRAME
        assert abs(region.end - _s(end_s)) <= FRAME


def test_cuts_exactly_at_planted_silences_with_padding():
    planted = [(0.5, 1.7), (2.3, 3.1), (3.9, 5.0)]
    data = speech_like(planted, total_s=5.5)
    cuts = segment_utterances(data, SR, EnergyVAD(CFG.energy), CFG)
    assert len(cuts) == 3
    for cut, (start_s, end_s) in zip(cuts, planted, strict=True):
        assert isinstance(cut, UtteranceCut)
        assert abs(cut.speech_start - _s(start_s)) <= FRAME
        assert abs(cut.speech_end - _s(end_s)) <= FRAME
        assert cut.start == cut.speech_start - PAD
        assert cut.end == cut.speech_end + PAD
        assert cut.pad_ok


def test_short_pause_does_not_split():
    data = speech_like([(0.5, 1.5), (1.7, 2.5)], total_s=3.0)  # lặng 200 ms
    cuts = segment_utterances(data, SR, EnergyVAD(CFG.energy), CFG)
    assert len(cuts) == 1
    assert abs(cuts[0].speech_end - _s(2.5)) <= FRAME


def test_silence_of_exactly_300_ms_splits():
    data = speech_like([(0.5, 1.5), (1.8, 2.5)], total_s=3.0)
    assert len(segment_utterances(data, SR, EnergyVAD(CFG.energy), CFG)) == 2


def test_off_grid_300_ms_silence_splits_with_sample_accurate_edges():
    # Biên lệch lưới khung 10 ms: khoảng lặng thật 300 ms không được đo thành 290 ms.
    planted = [(0.5, 1.505), (1.805, 2.6)]
    data = speech_like(planted, total_s=3.2)
    cuts = segment_utterances(data, SR, EnergyVAD(CFG.energy), CFG)
    assert len(cuts) == 2
    tolerance = round(0.001 * SR)
    for cut, (start_s, end_s) in zip(cuts, planted, strict=True):
        assert abs(cut.speech_start - _s(start_s)) <= tolerance
        assert abs(cut.speech_end - _s(end_s)) <= tolerance


def test_off_grid_290_ms_silence_does_not_split():
    data = speech_like([(0.5, 1.505), (1.795, 2.6)], total_s=3.2)
    assert len(segment_utterances(data, SR, EnergyVAD(CFG.energy), CFG)) == 1


def test_silence_of_250_ms_does_not_split():
    data = speech_like([(0.5, 1.5), (1.75, 2.5)], total_s=3.0)
    assert len(segment_utterances(data, SR, EnergyVAD(CFG.energy), CFG)) == 1


class _FixedVAD:
    """VAD giả trả vùng cố định, để test phần ghép/đệm độc lập với EnergyVAD."""

    def __init__(self, regions):
        self.regions = regions

    def detect(self, audio, sample_rate):
        return [SpeechRegion(a, b) for a, b in self.regions]


def test_long_take_with_little_silence_still_detected():
    # Câu 10 s, lặng 0,2 s mỗi đầu (mức tối thiểu đặc tả cho phép): chỉ 4% là lặng.
    data = speech_like([(0.2, 9.8)], total_s=10.0)
    [cut] = segment_utterances(data, SR, EnergyVAD(CFG.energy), CFG)
    assert abs(cut.speech_start - _s(0.2)) <= FRAME
    assert abs(cut.speech_end - _s(9.8)) <= FRAME
    assert cut.pad_ok


def test_padding_limited_by_file_edges_and_flagged():
    vad = _FixedVAD([(_s(0.04), _s(1.0))])
    [cut] = segment_utterances(np.zeros(_s(1.05)), SR, vad, CFG)
    assert (cut.start, cut.end) == (0, _s(1.05))
    assert not cut.pad_ok


def test_padding_split_between_close_neighbours():
    # Hai vùng cách nhau đúng 300 ms: mỗi bên được 150 ms, hai đoạn cắt chạm nhau.
    vad = _FixedVAD([(_s(0.5), _s(1.0)), (_s(1.3), _s(2.0))])
    first, second = segment_utterances(np.zeros(_s(3.0)), SR, vad, CFG)
    assert first.end == second.start == _s(1.15)


def test_no_speech_gives_no_cuts():
    assert segment_utterances(noise(2.0, -70.0), SR, EnergyVAD(CFG.energy), CFG) == []
    assert segment_utterances(np.zeros(SR), SR, EnergyVAD(CFG.energy), CFG) == []


def test_multichannel_input_rejected():
    with pytest.raises(ValueError, match="mono"):
        EnergyVAD(CFG.energy).detect(np.zeros((SR, 2)), SR)


def test_cut_file_writes_lossless_24_bit_segments(tmp_path):
    planted = [(0.5, 1.7), (2.3, 3.1)]
    src = write_wav(tmp_path / "take.wav", speech_like(planted, total_s=3.6))
    out = cut_file(src, tmp_path / "cuts", EnergyVAD(CFG.energy), CFG)
    assert [p.name for p, _ in out] == ["take_0001.wav", "take_0002.wav"]
    original, _ = sf.read(src, dtype="int32")
    for path, cut in out:
        assert sf.info(path).subtype == "PCM_24"
        piece, _ = sf.read(path, dtype="int32")
        assert np.array_equal(piece, original[cut.start : cut.end])


def test_cut_rerun_is_byte_identical(tmp_path):
    src = write_wav(tmp_path / "take.wav", speech_like([(0.5, 1.7), (2.3, 3.1)], total_s=3.6))
    first = cut_file(src, tmp_path / "a", EnergyVAD(CFG.energy), CFG)
    second = cut_file(src, tmp_path / "b", EnergyVAD(CFG.energy), CFG)
    assert [sha256(p) for p, _ in first] == [sha256(p) for p, _ in second]


def test_cli_cuts_file(tmp_path, capsys):
    src = write_wav(tmp_path / "take.wav", speech_like([(0.5, 1.7), (2.3, 3.1)], total_s=3.6))
    assert main([str(src), str(tmp_path / "cuts")]) == 0
    capsys.readouterr()
    assert len(list((tmp_path / "cuts").glob("*.wav"))) == 2


@pytest.mark.parametrize("regions", [[(100, 50)], [(0, 10**9)], [(0, 500), (400, 900)]])
def test_invalid_regions_from_detector_raise(regions):
    with pytest.raises(ValueError, match="VAD"):
        segment_utterances(np.zeros(SR), SR, _FixedVAD(regions), CFG)
