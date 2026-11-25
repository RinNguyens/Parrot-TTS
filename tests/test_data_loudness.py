"""Bước 2 pipeline: đo loudness ITU-R BS.1770 (pyloudnorm) và đưa về -23 LUFS.

API: ``vietvoice.data.loudness``: ``measure_loudness``, ``normalize``, ``normalize_file``, ``PeakLimitError``.
"""

import numpy as np
import pyloudnorm
import pytest
import soundfile as sf
from fixtures_audio import SR, noise, sha256, speech_like, write_wav

from vietvoice.data.loudness import (
    PeakLimitError,
    load_loudness_config,
    main,
    measure_loudness,
    normalize,
    normalize_file,
)


def _lufs(data, sr=SR):
    return pyloudnorm.Meter(sr).integrated_loudness(data)


def test_target_comes_from_yaml():
    assert load_loudness_config().target_lufs == -23.0


@pytest.mark.parametrize("speech_db", [-35.0, -28.0, -18.0])
def test_normalize_reaches_target(speech_db):
    data = speech_like([(0.5, 2.5), (3.0, 5.0)], total_s=5.5, speech_dbfs=speech_db)
    out, result = normalize(data, SR)
    assert _lufs(out) == pytest.approx(-23.0, abs=0.5)
    assert result.output_lufs == pytest.approx(-23.0, abs=0.5)
    assert result.input_lufs == pytest.approx(_lufs(data), abs=1e-6)
    assert result.gain_db == pytest.approx(result.output_lufs - result.input_lufs, abs=0.1)


def test_quiet_input_with_gated_blocks_still_reaches_target():
    # Rất nhỏ: một phần khối nằm dưới cổng tuyệt đối -70 LUFS trước khi tăng gain.
    data = speech_like([(0.5, 2.5)], total_s=4.0, speech_dbfs=-62.0, noise_dbfs=-110.0)
    out, _ = normalize(data, SR)
    assert _lufs(out) == pytest.approx(-23.0, abs=0.5)


def test_file_normalized_to_target_and_24_bit(tmp_path):
    src = write_wav(tmp_path / "in.wav", speech_like([(0.5, 2.5)], total_s=3.0, speech_dbfs=-30.0))
    dst = tmp_path / "out" / "out.wav"
    result = normalize_file(src, dst)
    info = sf.info(dst)
    assert (info.samplerate, info.channels, info.subtype, info.format) == (SR, 1, "PCM_24", "WAV")
    data, sr = sf.read(dst)
    assert _lufs(data, sr) == pytest.approx(-23.0, abs=0.5)
    assert result.output_lufs == pytest.approx(_lufs(data, sr), abs=0.05)


def test_refuses_when_gain_would_push_peak_over_limit():
    # Ồn rất nhỏ + một xung -6 dBFS: tăng lên -23 LUFS thì xung vượt -3 dBFS.
    data = noise(3.0, -55.0)
    data[SR] = 0.5
    with pytest.raises(PeakLimitError, match="dBFS"):
        normalize(data, SR)


def test_refusal_writes_no_file(tmp_path):
    data = noise(3.0, -55.0)
    data[SR] = 0.5
    src = write_wav(tmp_path / "in.wav", data)
    dst = tmp_path / "out.wav"
    with pytest.raises(PeakLimitError):
        normalize_file(src, dst)
    assert not dst.exists()


def test_silence_cannot_be_normalized():
    with pytest.raises(ValueError, match="loudness"):
        normalize(np.zeros(3 * SR), SR)


def test_too_short_to_measure():
    with pytest.raises(ValueError, match="ngắn"):
        measure_loudness(noise(0.2, -20.0), SR)


def test_rerun_is_byte_identical(tmp_path):
    src = write_wav(tmp_path / "in.wav", speech_like([(0.5, 2.5)], total_s=3.0, speech_dbfs=-30.0))
    normalize_file(src, tmp_path / "a.wav")
    normalize_file(src, tmp_path / "b.wav")
    assert sha256(tmp_path / "a.wav") == sha256(tmp_path / "b.wav")


def test_cli_normalizes_file(tmp_path, capsys):
    src = write_wav(tmp_path / "in.wav", speech_like([(0.5, 2.5)], total_s=3.0, speech_dbfs=-30.0))
    assert main([str(src), str(tmp_path / "out.wav")]) == 0
    assert "LUFS" in capsys.readouterr().out
    data, sr = sf.read(tmp_path / "out.wav")
    assert _lufs(data, sr) == pytest.approx(-23.0, abs=0.5)
