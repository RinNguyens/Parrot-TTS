"""Bước 12 + schema metadata (đặc tả mục 4): Utterance, Speaker, đọc/ghi JSONL có kiểm tra.

API: ``vietvoice.data.manifest``: ``Utterance``, ``Speaker``, ``write_manifest``, ``read_manifest``,
``read_speakers``, ``write_speakers``, ``check_consent``, ``ManifestError``.
"""

import json

import pytest
from fixtures_audio import sha256
from pydantic import ValidationError

from vietvoice.data.manifest import (
    ManifestError,
    Speaker,
    Utterance,
    check_consent,
    load_manifest_config,
    read_manifest,
    read_speakers,
    write_manifest,
    write_speakers,
)

# Dòng ví dụ trong đặc tả mục 4.
SPEC_LINE = (
    '{"id": "hn_f01_000123", "speaker_id": "hn_f01", "audio_path": "wavs/hn_f01/000123.wav", '
    '"duration_s": 4.82, "sample_rate": 48000, "text_raw": "Giá vàng hôm nay tăng 2%", '
    '"text_norm": "giá vàng hôm nay tăng hai phần trăm", '
    '"phonemes": "z a T5 _ v a ŋ T2 _ h o m T1 _ n a j T1 _ t a ŋ T1 _ h a j T1 _ f ə n T2 _ c a m T1", '
    '"tones": [5, 2, 1, 1, 1, 1, 2, 1], "dialect": "bac", "province": "HN", "style": "tin_tuc", '
    '"speech_rate": 5.6, "f0_mean_st": 0.0, "alignment_path": "align/hn_f01/000123.TextGrid", '
    '"snr_db": 47.3, "loudness_lufs": -23.0, "asr_cer": 0.012, "qc_status": "pass", "split": "train", '
    '"consent_id": "CT-2026-007", "license": "vietvoice-commercial-v1", "recorded_at": "2027-01-15"}'
)


def _row(**changes):
    row = json.loads(SPEC_LINE)
    row.update(changes)
    return row


def _bad(**changes):
    with pytest.raises(ValidationError):
        Utterance.model_validate(_row(**changes))


def test_spec_example_line_is_valid():
    utt = Utterance.model_validate_json(SPEC_LINE)
    assert utt.id == "hn_f01_000123"
    assert utt.emotion is None
    assert str(utt.recorded_at) == "2027-01-15"


def test_optional_emotion_enum():
    assert Utterance.model_validate(_row(emotion="vui")).emotion == "vui"
    _bad(emotion="gian_du")


@pytest.mark.parametrize(
    "changes",
    [
        {"dialect": "trung"},
        {"style": "rap"},
        {"qc_status": "ok"},
        {"split": "dev"},
        {"tones": [5, 2, 1, 1, 1, 1, 2, 9]},
        {"tones": [5, 2, 1, 1, 1, 1, 2, 0]},
        {"sample_rate": 44100},
        {"duration_s": 0.0},
        {"asr_cer": -0.1},
        {"text_norm": ""},
        {"snr_db": float("nan")},
        {"recorded_at": "15/01/2027"},
        {"extra_field": 1},
    ],
)
def test_bad_values_rejected(changes):
    _bad(**changes)


def test_missing_required_field_rejected():
    row = _row()
    del row["consent_id"]
    with pytest.raises(ValidationError):
        Utterance.model_validate(row)


def test_hue_dialect_allowed():
    assert Utterance.model_validate(_row(dialect="hue")).dialect == "hue"


@pytest.mark.parametrize("bad_id", ["hn_f01_12345", "hn_f01_0001234", "hn_f02_000123", "hn_f01-000123", "000123"])
def test_id_must_be_speaker_plus_six_digits(bad_id):
    _bad(id=bad_id)


def test_bad_speaker_id_rejected():
    _bad(speaker_id="HN F01", id="HN F01_000123")


@pytest.mark.parametrize("path", ["/abs/wavs/x.wav", "../wavs/x.wav", "wavs/../../x.wav", ""])
def test_paths_must_be_relative_inside_data(path):
    _bad(audio_path=path)
    _bad(alignment_path=path)


@pytest.mark.parametrize("duration", [0.5, 15.5])
def test_duration_out_of_range_cannot_pass(duration):
    _bad(duration_s=duration, qc_status="pass")
    assert Utterance.model_validate(_row(duration_s=duration, qc_status="review")).qc_status == "review"


def test_duration_bounds_are_inclusive():
    for duration in (1.0, 15.0):
        Utterance.model_validate(_row(duration_s=duration))


@pytest.mark.parametrize("changes", [{"snr_db": 5.0}, {"snr_db": 39.9}, {"asr_cer": 0.9}, {"asr_cer": 0.051}])
def test_pass_requires_qa_snr_and_asr_cer(changes):
    _bad(qc_status="pass", **changes)
    assert Utterance.model_validate(_row(qc_status="review", **changes)).qc_status == "review"


def test_pass_limits_inclusive_and_from_config():
    cfg = load_manifest_config()
    assert cfg.max_asr_cer == 0.05
    assert cfg.min_snr_db == 40.0  # lấy từ qa.yaml
    Utterance.model_validate(_row(snr_db=40.0, asr_cer=0.05))


def test_non_commercial_license_rejected_in_train():
    with pytest.raises(ValidationError, match="license"):
        Utterance.model_validate(_row(license="cc-by-nc-4.0", split="train"))


def test_non_commercial_license_allowed_outside_train():
    assert Utterance.model_validate(_row(license="cc-by-nc-4.0", split="test")).split == "test"


def test_train_license_allow_list_from_yaml():
    assert "vietvoice-commercial-v1" in load_manifest_config().train_licenses


def test_tones_length_must_match_syllables():
    with pytest.raises(ValidationError, match="tones"):
        Utterance.model_validate(_row(tones=[5, 2, 1]))


def test_phoneme_tone_token_out_of_range_rejected():
    _bad(phonemes="z a T9", tones=[1])


def test_roundtrip_jsonl(tmp_path):
    utts = [Utterance.model_validate(_row()), Utterance.model_validate(_row(id="hn_f01_000124", emotion="vui"))]
    path = tmp_path / "train.jsonl"
    write_manifest(utts, path)
    assert read_manifest(path) == utts
    assert path.read_text(encoding="utf-8").count("\n") == 2
    assert "Giá vàng" in path.read_text(encoding="utf-8")


def test_write_is_byte_identical(tmp_path):
    utts = [Utterance.model_validate(_row())]
    write_manifest(utts, tmp_path / "a.jsonl")
    write_manifest(utts, tmp_path / "b.jsonl")
    assert sha256(tmp_path / "a.jsonl") == sha256(tmp_path / "b.jsonl")


def test_bad_line_error_names_line_number(tmp_path):
    path = tmp_path / "train.jsonl"
    good = json.dumps(_row(), ensure_ascii=False)
    bad = json.dumps(_row(id="hn_f01_000124", dialect="trung"), ensure_ascii=False)
    path.write_text(f"{good}\n{bad}\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="dòng 2"):
        read_manifest(path)


def test_invalid_json_line_error_names_line_number(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_row(), ensure_ascii=False) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="dòng 2"):
        read_manifest(path)


def test_blank_line_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\n" + json.dumps(_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="dòng 1"):
        read_manifest(path)


def test_duplicate_id_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    line = json.dumps(_row(), ensure_ascii=False)
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="dòng 2"):
        read_manifest(path)


def test_write_refuses_duplicate_ids(tmp_path):
    utt = Utterance.model_validate(_row())
    with pytest.raises(ManifestError, match="trùng"):
        write_manifest([utt, utt], tmp_path / "x.jsonl")


SPEAKER = {
    "speaker_id": "hn_f01",
    "voice_name": "Mai (Hà Nội)",
    "gender": "nữ",
    "age_range": "25–34",
    "dialect": "bac",
    "province": "HN",
    "grew_up_in": "Hà Nội, ≥ 15 năm",
    "voice_traits": "ấm, trung, rõ",
    "f0_baseline_hz": 215,
    "mic": "Neumann TLM 103",
    "room": "phòng thu A",
    "consent_id": "CT-2026-007",
    "rights": ["thương mại", "huấn luyện", "phân phối giọng tổng hợp"],
    "hours_recorded": 12.4,
}


def test_speaker_spec_example_valid():
    assert Speaker.model_validate(SPEAKER).speaker_id == "hn_f01"


@pytest.mark.parametrize(
    "changes",
    [{"dialect": "trung"}, {"f0_baseline_hz": -1}, {"hours_recorded": -0.5}, {"rights": []}, {"consent_id": ""}],
)
def test_bad_speaker_rejected(changes):
    with pytest.raises(ValidationError):
        Speaker.model_validate({**SPEAKER, **changes})


def test_speakers_json_roundtrip_and_duplicates(tmp_path):
    path = tmp_path / "speakers.json"
    speakers = [Speaker.model_validate(SPEAKER), Speaker.model_validate({**SPEAKER, "speaker_id": "hn_m01"})]
    write_speakers(speakers, path)
    assert read_speakers(path) == speakers
    path.write_text(json.dumps([SPEAKER, SPEAKER], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ManifestError, match="trùng"):
        read_speakers(path)


def test_speakers_bad_entry_names_index(tmp_path):
    path = tmp_path / "speakers.json"
    path.write_text(json.dumps([SPEAKER, {**SPEAKER, "dialect": "x"}], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ManifestError, match="mục 2"):
        read_speakers(path)


def test_consent_and_speaker_must_match():
    speaker = Speaker.model_validate(SPEAKER)
    check_consent([Utterance.model_validate(_row())], [speaker])
    with pytest.raises(ValueError, match="consent_id"):
        check_consent([Utterance.model_validate(_row(consent_id="CT-2026-999"))], [speaker])
    with pytest.raises(ValueError, match="speakers.json"):
        check_consent([Utterance.model_validate(_row())], [])
    with pytest.raises(ValueError, match="dialect"):
        check_consent([Utterance.model_validate(_row(dialect="nam"))], [speaker])
