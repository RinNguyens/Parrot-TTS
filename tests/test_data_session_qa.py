"""P4: QA ngay sau mỗi phiên thu.

Ngưỡng (đặc tả mục 5): đúng định dạng, SNR >= 40 dB, đỉnh <= -3 dBFS, không clip, nền ồn <= -60 dBFS.
API: ``parrot_tts.data.session_qa``: ``measure_levels``, ``check_file``, ``run_session``, ``main``.
"""

import json

import numpy as np
import pytest
from fixtures_audio import SR, good_take, noise, sha256, sine, speech_like, write_wav

from parrot_tts.data.session_qa import check_file, load_qa_config, main, measure_levels, run_session


@pytest.mark.parametrize(("speech_db", "noise_db"), [(-20.0, -70.0), (-25.0, -65.0), (-15.0, -80.0)])
def test_snr_and_noise_floor_match_planted_levels(speech_db, noise_db):
    data = speech_like([(0.5, 2.0), (2.5, 4.0)], total_s=4.5, speech_dbfs=speech_db, noise_dbfs=noise_db)
    levels = measure_levels(data, SR)
    assert levels.noise_floor_dbfs == pytest.approx(noise_db, abs=1.0)
    assert levels.snr_db == pytest.approx(speech_db - noise_db, abs=1.0)


def test_long_take_with_minimum_silence_measured_correctly():
    # Câu 10 s, lặng 0,2 s mỗi đầu: chỉ 4% thời lượng là lặng.
    levels = measure_levels(speech_like([(0.2, 9.8)], total_s=10.0), SR)
    assert levels.noise_floor_dbfs == pytest.approx(-70.0, abs=1.0)
    assert levels.snr_db == pytest.approx(50.0, abs=1.0)


def test_peak_is_sample_peak_in_dbfs():
    data = speech_like([(0.5, 1.5)], total_s=2.0, speech_dbfs=-20.0)
    data[100] = 0.5
    assert measure_levels(data, SR).peak_dbfs == pytest.approx(20 * np.log10(0.5), abs=1e-3)


def test_no_speech_frames_gives_no_snr():
    levels = measure_levels(noise(2.0, -70.0), SR)
    assert levels.snr_db is None


def test_digital_silence_is_clamped_to_db_floor():
    floor = load_qa_config().analysis.db_floor
    levels = measure_levels(np.zeros(SR), SR)
    assert levels.noise_floor_dbfs == pytest.approx(floor)
    assert levels.peak_dbfs == pytest.approx(floor)
    assert levels.snr_db is None
    assert levels.longest_zero_run_s == pytest.approx(1.0)


def test_digital_silence_file_rejected_with_zero_run_reason(tmp_path):
    result = check_file(write_wav(tmp_path / "a.wav", np.zeros(SR)))
    assert not result.passed
    assert any("lặng tuyệt đối" in r for r in result.reasons)


def test_digital_zero_lead_in_does_not_hide_noise(tmp_path):
    # 100 ms số 0 tuyệt đối trước một câu có nền ồn -45 dBFS: nền ồn phải đo ra ~-45, không phải -150.
    take = speech_like([(0.3, 2.5)], total_s=3.0, noise_dbfs=-45.0)
    data = np.concatenate([np.zeros(round(0.1 * SR)), take])
    levels = measure_levels(data, SR)
    assert levels.noise_floor_dbfs == pytest.approx(-45.0, abs=1.0)
    result = check_file(write_wav(tmp_path / "a.wav", data))
    assert not result.passed
    assert any("nền ồn" in r for r in result.reasons)
    assert any("lặng tuyệt đối" in r for r in result.reasons)


def test_short_zero_runs_in_noise_are_not_reported(tmp_path):
    data = speech_like([(0.5, 2.0)], total_s=2.5)
    data[SR : SR + 20] = 0.0
    assert check_file(write_wav(tmp_path / "a.wav", data)).passed


@pytest.mark.parametrize("total_s", [1.0, 3.0])
def test_trailing_partial_frame_does_not_bias_noise_floor(total_s):
    take = speech_like([(0.2, total_s - 0.1)], total_s=total_s, speech_dbfs=-15.0, noise_dbfs=-58.5)
    with_tail = np.concatenate([take, [0.0]])
    assert measure_levels(with_tail, SR).noise_floor_dbfs == pytest.approx(
        measure_levels(take, SR).noise_floor_dbfs, abs=0.01
    )
    assert measure_levels(with_tail, SR).noise_floor_dbfs > -60.0


def test_file_shorter_than_one_frame_rejected(tmp_path):
    result = check_file(write_wav(tmp_path / "a.wav", noise(0.005, -70.0)))
    assert not result.passed
    assert any("ngắn" in r for r in result.reasons)


def test_good_take_passes(tmp_path):
    result = check_file(good_take(tmp_path / "a.wav"))
    assert result.passed, result.reasons
    assert result.snr_db is not None and result.snr_db >= 40


def test_low_snr_rejected(tmp_path):
    data = speech_like([(0.5, 2.0)], total_s=2.5, speech_dbfs=-20.0, noise_dbfs=-55.0)
    result = check_file(write_wav(tmp_path / "a.wav", data))
    assert not result.passed
    assert any("SNR" in r for r in result.reasons)
    assert any("nền ồn" in r for r in result.reasons)


def test_noise_floor_rejected_even_if_snr_ok(tmp_path):
    # Tiếng -8 dBFS, ồn -55 dBFS: SNR 47 dB đạt nhưng nền ồn quá -60 dBFS.
    data = speech_like([(0.5, 2.0)], total_s=2.5, speech_dbfs=-8.0, noise_dbfs=-55.0)
    result = check_file(write_wav(tmp_path / "a.wav", data))
    assert not result.passed
    assert not any("SNR" in r for r in result.reasons)
    assert any("nền ồn" in r for r in result.reasons)


def test_hot_peak_rejected(tmp_path):
    data = speech_like([(0.5, 2.0)], total_s=2.5, speech_dbfs=-4.0)
    result = check_file(write_wav(tmp_path / "a.wav", data))
    assert not result.passed
    assert any("đỉnh" in r for r in result.reasons)


def test_clipped_take_rejected(tmp_path):
    data = speech_like([(0.5, 2.0)], total_s=2.5, speech_dbfs=-20.0)
    data[SR : SR + 2000] = np.clip(data[SR : SR + 2000] * 1000, -1.0, 1.0)
    result = check_file(write_wav(tmp_path / "a.wav", data))
    assert not result.passed
    assert result.clipped_runs > 0
    assert any("clip" in r for r in result.reasons)


def test_wrong_format_rejected(tmp_path):
    data = speech_like([(0.5, 2.0)], total_s=2.5)
    result = check_file(write_wav(tmp_path / "a.wav", data, subtype="PCM_16"))
    assert not result.passed
    assert not result.format_ok


def test_thresholds_come_from_yaml():
    cfg = load_qa_config()
    assert (cfg.qa.min_snr_db, cfg.qa.max_peak_dbfs, cfg.qa.max_noise_floor_dbfs) == (40.0, -3.0, -60.0)


def _session(tmp_path):
    session = tmp_path / "raw" / "hn_f01" / "s01"
    good_take(session / "001.wav", seed=1)
    good_take(session / "002.wav", seed=2)
    good_take(session / "003.wav", seed=3)
    write_wav(session / "004.wav", sine(2.0, -20.0, sr=44_100), sr=44_100)
    (session / "notes.txt").write_text("phiếu ghi phiên", encoding="utf-8")
    return session


def test_session_summary_and_reject_rate(tmp_path):
    summary = run_session(_session(tmp_path))
    assert [f.name for f in summary.files] == ["001.wav", "002.wav", "003.wav", "004.wav"]
    assert (summary.total, summary.passed, summary.failed) == (4, 3, 1)
    assert summary.reject_rate == pytest.approx(0.25)


def test_empty_session_raises(tmp_path):
    with pytest.raises(ValueError, match="không có file"):
        run_session(tmp_path)


def test_missing_session_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_session(tmp_path / "nope")


def test_cli_exit_code_table_and_json(tmp_path, capsys):
    out = tmp_path / "qa.json"
    code = main([str(_session(tmp_path)), "--json", str(out)])
    assert code == 1
    printed = capsys.readouterr().out
    assert "004.wav" in printed and "loại" in printed and "25.0%" in printed
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["failed"] == 1
    assert report["files"][3]["passed"] is False


def test_cli_exit_zero_when_all_pass(tmp_path, capsys):
    session = tmp_path / "s"
    good_take(session / "001.wav")
    assert main([str(session)]) == 0
    capsys.readouterr()


def test_json_report_is_reproducible(tmp_path, capsys):
    session = _session(tmp_path)
    main([str(session), "--json", str(tmp_path / "1.json")])
    main([str(session), "--json", str(tmp_path / "2.json")])
    capsys.readouterr()
    assert sha256(tmp_path / "1.json") == sha256(tmp_path / "2.json")


def test_cli_bad_session_dir_exits_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "nope")])
    assert exc.value.code == 2
    assert "không thấy thư mục" in capsys.readouterr().err
