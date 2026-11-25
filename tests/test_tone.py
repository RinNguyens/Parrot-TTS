"""Thanh điệu theo chữ viết (chưa áp luật vùng).

Quy ước số thanh theo doc đặc tả, mục 3:
T1 ngang, T2 huyền, T3 ngã, T4 hỏi, T5 sắc, T6 nặng,
T7 sắc trong âm tiết tắc (-p -t -c -ch), T8 nặng trong âm tiết tắc.

API cần có: ``parrot_tts.text.g2p.syllable_tone(syllable: str) -> int``.
"""

import unicodedata

import pytest

from parrot_tts.text.g2p import syllable_tone


@pytest.mark.parametrize(
    ("syllable", "tone"),
    [("ma", 1), ("mà", 2), ("mã", 3), ("mả", 4), ("má", 5), ("mạ", 6)],
)
def test_six_tones_on_open_syllable(syllable, tone):
    assert syllable_tone(syllable) == tone


@pytest.mark.parametrize(
    ("syllable", "tone"),
    [
        ("mát", 7),
        ("tháp", 7),
        ("các", 7),
        ("sách", 7),
        ("mạt", 8),
        ("đẹp", 8),
        ("học", 8),
        ("lịch", 8),
    ],
)
def test_checked_syllables_use_tone_7_and_8(syllable, tone):
    assert syllable_tone(syllable) == tone


@pytest.mark.parametrize(("syllable", "tone"), [("ăn", 1), ("cơm", 1), ("uống", 5), ("anh", 1)])
def test_non_checked_syllables_keep_six_tone_numbers(syllable, tone):
    assert syllable_tone(syllable) == tone


@pytest.mark.parametrize("syllable", ["hoà", "thuý", "mã", "Việt", "người"])
def test_composed_and_decomposed_unicode_give_same_tone(syllable):
    nfc = unicodedata.normalize("NFC", syllable)
    nfd = unicodedata.normalize("NFD", syllable)
    assert syllable_tone(nfc) == syllable_tone(nfd)


@pytest.mark.parametrize(("old", "new"), [("hoà", "hòa"), ("thuý", "thúy"), ("khoẻ", "khỏe")])
def test_old_and_new_tone_mark_placement_are_equal(old, new):
    assert syllable_tone(old) == syllable_tone(new)


@pytest.mark.parametrize(("upper", "lower"), [("MÃ", "mã"), ("Học", "học"), ("ĐẸP", "đẹp")])
def test_case_does_not_change_tone(upper, lower):
    assert syllable_tone(upper) == syllable_tone(lower)
