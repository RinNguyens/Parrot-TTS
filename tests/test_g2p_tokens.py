"""Chuỗi token đầu ra của G2P: định dạng ở doc đặc tả, mục 3.

API cần có: ``parrot_tts.text.g2p.g2p(text: str, dialect: str) -> list[str]``.
- Mỗi âm tiết kết thúc bằng đúng một token thanh T1..T8.
- ``_`` là ranh giới từ; dấu , . ? ! giữ nguyên thành token riêng.
- ``ʔ`` đứng đầu âm tiết mở đầu bằng nguyên âm.
"""

import re

import pytest

from parrot_tts.text.g2p import g2p

TONE = re.compile(r"^T[1-8]$")
DIALECTS = ["bac", "nam"]
CHECKED_CODAS = {"p", "t", "k", "c", "k͡p", "ʲk"}

SENTENCES = [
    "ăn cơm chưa",
    "giá vàng hôm nay tăng mạnh",
    "học sinh đọc sách ở thư viện",
    "người Việt nói tiếng Việt",
    "quả xoài này ngọt lắm",
]


def syllables(tokens: list[str]) -> list[list[str]]:
    """Tách chuỗi token thành từng âm tiết, mỗi âm tiết kết thúc ở token thanh."""
    out, current = [], []
    for tok in tokens:
        if tok in {"_", ",", ".", "?", "!"}:
            continue
        current.append(tok)
        if TONE.match(tok):
            out.append(current)
            current = []
    assert not current, f"còn token không có thanh ở cuối: {current}"
    return out


def test_doc_example_an_com_bac():
    assert g2p("ăn cơm", "bac") == ["ʔ", "a", "n", "T1", "_", "k", "ə", "m", "T1"]


def test_doc_example_anh_nam():
    assert g2p("anh", "nam") == ["ʔ", "a", "n", "T1"]


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("sentence", SENTENCES)
def test_one_tone_token_per_syllable(sentence, dialect):
    tokens = g2p(sentence, dialect)
    assert len(syllables(tokens)) == len(sentence.split())


@pytest.mark.parametrize("dialect", DIALECTS)
def test_word_boundaries(dialect):
    tokens = g2p("ăn cơm chưa", dialect)
    assert tokens.count("_") == 2


@pytest.mark.parametrize("dialect", DIALECTS)
def test_punctuation_is_kept_as_tokens(dialect):
    tokens = g2p("ăn cơm chưa? chưa, ăn đi!", dialect)
    for mark in ["?", ",", "!"]:
        assert mark in tokens


@pytest.mark.parametrize("sentence", SENTENCES + ["mã mả mạ mỡ mở"])
def test_southern_never_uses_tone_3(sentence):
    assert "T3" not in g2p(sentence, "nam")


def test_northern_keeps_nga_and_hoi_apart():
    tones = [s[-1] for s in syllables(g2p("mã mả", "bac"))]
    assert tones == ["T3", "T4"]


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("sentence", SENTENCES)
def test_checked_syllables_only_take_tone_7_or_8(sentence, dialect):
    for syl in syllables(g2p(sentence, dialect)):
        coda = syl[-2]
        if coda in CHECKED_CODAS:
            assert syl[-1] in {"T7", "T8"}, syl


@pytest.mark.parametrize("dialect", DIALECTS)
def test_output_is_lowercase(dialect):
    tokens = g2p("Người Việt NÓI", dialect)
    assert all(t == t.lower() or TONE.match(t) for t in tokens)


def test_same_text_differs_between_dialects():
    assert g2p("da", "bac") != g2p("da", "nam")


def test_unknown_dialect_is_rejected():
    with pytest.raises(ValueError):
        g2p("ăn cơm", "hanoi")
