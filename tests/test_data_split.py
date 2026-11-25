"""Bước 12: chia train/val/test = 94/3/3 theo người nói và văn phong, câu test không vào train.

API: ``vietvoice.data.manifest.split_utterances(utterances, test_texts, config=None) -> SplitResult``.
"""

import itertools

import pytest

from vietvoice.data.manifest import Utterance, load_manifest_config, split_utterances

SPEAKERS = ["hn_f01", "hn_m01", "sg_f01", "sg_m01"]
STYLES = ["tin_tuc", "ke_chuyen", "quang_cao", "hoi_thoai"]


def _utt(speaker: str, n: int, text: str, style: str = "tin_tuc", **changes) -> Utterance:
    row = {
        "id": f"{speaker}_{n:06d}",
        "speaker_id": speaker,
        "audio_path": f"wavs/{speaker}/{n:06d}.wav",
        "duration_s": 3.0,
        "sample_rate": 48000,
        "text_raw": text,
        "text_norm": text,
        "phonemes": "m a T1",
        "tones": [1],
        "dialect": "bac" if speaker.startswith("hn") else "nam",
        "province": "HN" if speaker.startswith("hn") else "SG",
        "style": style,
        "speech_rate": 5.0,
        "f0_mean_st": 0.0,
        "alignment_path": f"align/{speaker}/{n:06d}.TextGrid",
        "snr_db": 45.0,
        "loudness_lufs": -23.0,
        "asr_cer": 0.0,
        "qc_status": "pass",
        "split": "train",
        "consent_id": "CT-1",
        "license": "vietvoice-commercial-v1",
        "recorded_at": "2027-01-15",
    }
    row.update(changes)
    return Utterance.model_validate(row)


def _corpus(per_stratum: int = 200) -> list[Utterance]:
    utts = []
    for speaker, style in itertools.product(SPEAKERS, STYLES):
        base = len([u for u in utts if u.speaker_id == speaker])
        for i in range(per_stratum):
            utts.append(_utt(speaker, base + i, f"câu {speaker} {style} số {i}", style))
    return utts


def test_ratios_from_yaml():
    cfg = load_manifest_config()
    assert (cfg.split.train, cfg.split.val, cfg.split.test) == (0.94, 0.03, 0.03)


def test_proportions_overall_and_per_stratum():
    result = split_utterances(_corpus(), test_texts=set())
    total = len(result.utterances)
    assert total == 16 * 200
    counts = {s: sum(u.split == s for u in result.utterances) for s in ("train", "val", "test")}
    assert counts["train"] / total == pytest.approx(0.94, abs=0.005)
    assert counts["val"] / total == pytest.approx(0.03, abs=0.005)
    assert counts["test"] / total == pytest.approx(0.03, abs=0.005)
    for speaker, style in itertools.product(SPEAKERS, STYLES):
        stratum = [u for u in result.utterances if (u.speaker_id, u.style) == (speaker, style)]
        assert sum(u.split == "val" for u in stratum) == 6
        assert sum(u.split == "test" for u in stratum) == 6


def test_deterministic_with_seed_and_independent_of_input_order():
    utts = _corpus(50)
    first = split_utterances(utts, test_texts=set())
    second = split_utterances(list(reversed(utts)), test_texts=set())
    assert first == second
    assert [u.id for u in first.utterances] == sorted(u.id for u in utts)


def test_seed_changes_assignment():
    utts = _corpus(50)
    cfg = load_manifest_config()
    other = cfg.model_copy(update={"split": cfg.split.model_copy(update={"seed": cfg.split.seed + 1})})
    a = split_utterances(utts, test_texts=set())
    b = split_utterances(utts, test_texts=set(), config=other)
    assert [u.split for u in a.utterances] != [u.split for u in b.utterances]


def test_test_sentences_never_in_train():
    utts = _corpus(50)
    tone_test = {u.text_norm for u in utts[::7]}
    result = split_utterances(utts, test_texts=tone_test)
    for utt in result.utterances:
        if utt.text_norm in tone_test:
            assert utt.split != "train"


def test_test_sentence_match_ignores_case_and_spacing():
    utts = [_utt("hn_f01", i, f"câu số {i}") for i in range(40)]
    result = split_utterances(utts, test_texts={"  CÂU   SỐ 3 "})
    assert next(u for u in result.utterances if u.id == "hn_f01_000003").split != "train"


def test_test_sentence_matched_on_raw_text_too():
    utts = [_utt("hn_f01", i, f"câu số {i}", text_raw=f"Câu số {i}!") for i in range(40)]
    result = split_utterances(utts, test_texts={"Câu số 5!"})
    assert next(u for u in result.utterances if u.id == "hn_f01_000005").split != "train"


def test_test_sentence_flagged_by_one_speaker_applies_to_all():
    # Cùng text_norm, chỉ text_raw của người nói thứ hai nằm trong bộ câu test.
    utts = [_utt("hn_f01", i, f"câu {i}", text_raw=f"Câu {i}.") for i in range(30)]
    utts += [_utt("sg_f01", i, f"câu {i}", text_raw=f"Câu {i}!") for i in range(30)]
    result = split_utterances(utts, test_texts={f"Câu {i}!" for i in range(30)})
    assert all(u.split == "test" for u in result.utterances)


def test_same_script_line_across_dialects_stays_in_one_split():
    # Cùng câu kịch bản, chuẩn hoá khác nhau theo vùng ("nghìn" / "ngàn").
    utts = []
    for i in range(200):
        raw = f"Giá {100 + i} nghìn"
        utts.append(_utt("hn_f01", i, f"giá {i} nghìn", text_raw=raw))
        utts.append(_utt("sg_f01", i, f"giá {i} ngàn", text_raw=raw))
    result = split_utterances(utts, test_texts=set())
    by_raw: dict[str, set[str]] = {}
    for utt in result.utterances:
        by_raw.setdefault(utt.text_raw, set()).add(utt.split)
    assert all(len(splits) == 1 for splits in by_raw.values())
    assert {s for splits in by_raw.values() for s in splits} == {"train", "val", "test"}


def test_same_text_never_crosses_splits():
    utts = []
    for speaker in SPEAKERS:
        utts += [_utt(speaker, i, f"câu chung {i}") for i in range(100)]
    result = split_utterances(utts, test_texts=set())
    by_text: dict[str, set[str]] = {}
    for utt in result.utterances:
        by_text.setdefault(utt.text_norm, set()).add(utt.split)
    assert all(len(splits) == 1 for splits in by_text.values())


def test_duplicate_text_of_same_speaker_dropped():
    utts = [_utt("hn_f01", i, f"câu {i}") for i in range(20)]
    utts += [_utt("hn_f01", 100, "câu 3"), _utt("hn_f01", 101, "Câu  3")]
    result = split_utterances(utts, test_texts=set())
    assert result.dropped_duplicates == ("hn_f01_000100", "hn_f01_000101")
    assert len(result.utterances) == 20


def test_duplicate_keeps_pass_over_review():
    utts = [_utt("hn_f01", 1, "câu", qc_status="review"), _utt("hn_f01", 2, "câu")]
    result = split_utterances(utts, test_texts=set())
    assert [u.id for u in result.utterances] == ["hn_f01_000002"]
    assert result.dropped_duplicates == ("hn_f01_000001",)


def test_non_commercial_license_refused():
    utts = [_utt("hn_f01", i, f"câu {i}") for i in range(10)]
    utts.append(_utt("hn_f01", 99, "câu nc", license="cc-by-nc-4.0", split="test"))
    with pytest.raises(ValueError, match="hn_f01_000099"):
        split_utterances(utts, test_texts=set())


def test_empty_input_raises():
    with pytest.raises(ValueError, match="rỗng"):
        split_utterances([], test_texts=set())


def test_duplicate_ids_raise():
    utt = _utt("hn_f01", 1, "câu")
    with pytest.raises(ValueError, match="trùng"):
        split_utterances([utt, utt], test_texts=set())
