"""Bước 1 pipeline: kiểm tra định dạng (48 kHz, 24-bit PCM, mono, WAV) và clip.

API: ``vietvoice.data.ingest.check_format(path, config=None) -> FormatCheck``.
"""

import numpy as np
import pytest
import soundfile as sf
from fixtures_audio import SR, good_take, sine, write_wav

from vietvoice.data.ingest import FormatCheck, check_format, clipped_runs, load_ingest_config


def test_good_file_passes(tmp_path):
    result = check_format(good_take(tmp_path / "a.wav"))
    assert isinstance(result, FormatCheck)
    assert result.ok
    assert result.reasons == ()
    assert (result.sample_rate, result.channels, result.subtype) == (SR, 1, "PCM_24")
    assert result.clipped_runs == 0


def test_wrong_sample_rate_rejected(tmp_path):
    path = write_wav(tmp_path / "a.wav", sine(1.0, -20, sr=44_100), sr=44_100)
    result = check_format(path)
    assert not result.ok
    assert any("44100" in r for r in result.reasons)


def test_16_bit_rejected(tmp_path):
    path = write_wav(tmp_path / "a.wav", sine(1.0, -20), subtype="PCM_16")
    result = check_format(path)
    assert not result.ok
    assert any("PCM_16" in r for r in result.reasons)


def test_float_rejected(tmp_path):
    path = write_wav(tmp_path / "a.wav", sine(1.0, -20), subtype="FLOAT")
    assert not check_format(path).ok


def test_stereo_rejected(tmp_path):
    path = tmp_path / "a.wav"
    data = np.stack([sine(1.0, -20), sine(1.0, -20)], axis=1)
    sf.write(path, data, SR, subtype="PCM_24")
    result = check_format(path)
    assert not result.ok
    assert result.channels == 2
    assert any("mono" in r for r in result.reasons)


def test_non_wav_container_rejected(tmp_path):
    path = tmp_path / "a.flac"
    sf.write(path, sine(1.0, -20), SR, subtype="PCM_24", format="FLAC")
    result = check_format(path)
    assert not result.ok
    assert any("WAV" in r for r in result.reasons)


def test_several_problems_all_reported(tmp_path):
    path = tmp_path / "a.wav"
    data = np.stack([sine(1.0, -20, sr=44_100)] * 2, axis=1)
    sf.write(path, data, 44_100, subtype="PCM_16")
    assert len(check_format(path).reasons) == 3


def test_clipped_file_rejected(tmp_path):
    # Sin biên độ 2 bị cắt ở full scale -> mỗi đỉnh là một đoạn dài mẫu bằng full scale.
    path = write_wav(tmp_path / "a.wav", np.clip(2.0 * sine(1.0, -3.01), -1.0, 1.0))
    result = check_format(path)
    assert not result.ok
    assert result.clipped_runs > 0
    assert result.max_clip_run >= load_ingest_config().clipping.min_run
    assert any("clip" in r for r in result.reasons)


def test_single_full_scale_sample_is_not_a_clip_run(tmp_path):
    data = sine(1.0, -20)
    data[1000] = 1.0
    result = check_format(write_wav(tmp_path / "a.wav", data))
    assert result.clipped_runs == 0
    assert result.ok


def test_clip_within_one_lsb_counts():
    full = 2**23 - 1
    samples = np.array([0, full - 1, full, full - 1, 0, -(2**23) + 1, -(2**23), -(2**23) + 1, 0], dtype=np.int64)
    runs = clipped_runs(samples, bits=24, lsb_tolerance=1)
    assert runs == [3, 3]


def test_two_lsb_below_full_scale_is_not_clip():
    full = 2**23 - 1
    samples = np.array([full - 2] * 5, dtype=np.int64)
    assert clipped_runs(samples, bits=24, lsb_tolerance=1) == []


def test_unreadable_file_is_a_failure_with_reason(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not a wav file at all")
    result = check_format(path)
    assert not result.ok
    assert any("không đọc được" in r for r in result.reasons)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_format(tmp_path / "missing.wav")


def test_empty_file_rejected(tmp_path):
    path = write_wav(tmp_path / "a.wav", np.zeros(0))
    result = check_format(path)
    assert not result.ok
    assert any("không có mẫu" in r for r in result.reasons)


def test_wavex_24_bit_container_accepted(tmp_path):
    # WAV 24-bit từ nhiều phần mềm thu dùng WAVE_FORMAT_EXTENSIBLE, soundfile báo "WAVEX".
    path = tmp_path / "a.wav"
    sf.write(path, sine(1.0, -20), SR, subtype="PCM_24", format="WAVEX")
    assert sf.info(str(path)).format == "WAVEX"
    assert check_format(path).ok


def test_wavex_still_checks_bit_depth(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, sine(1.0, -20), SR, subtype="PCM_16", format="WAVEX")
    assert not check_format(path).ok
