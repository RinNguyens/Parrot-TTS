"""Phần không cần GPU của gói chạy local (src/parrot_tts/tts.py, __init__.py)."""

from __future__ import annotations

import json

import pytest

from parrot_tts import main
from parrot_tts import tts as T


def test_tts_text_keeps_plain_sentence_and_normalizes_numbers():
    assert T.tts_text("Xin chào  các bạn.", "bac") == "Xin chào các bạn."
    assert T.tts_text("Giảm 50% hôm nay.", "bac") == "giảm năm mươi phần trăm hôm nay."


def test_split_text_limits_syllables():
    long = ", ".join(["một hai ba bốn năm"] * 10) + "."
    chunks = T.split_text("Câu một. " + long)
    assert all(len(c.split()) <= T.MAX_SYL for c in chunks)
    assert " ".join(chunks).split() == ("Câu một. " + long).split()


def test_length_ok():
    assert T.length_ok(int(24000 * 0.505 * 4), 24000, "một hai ba bốn")[0]
    assert not T.length_ok(100, 24000, "một hai ba bốn")[0]


def test_list_voices(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PARROT_TTS_HOME", str(tmp_path))
    d = tmp_path / "voices" / "ha_my"
    d.mkdir(parents=True)
    (d / "prompt.wav").write_bytes(b"")
    (d / "voice.json").write_text(json.dumps({"text": "Chào.", "region": "nam", "desc": "nữ, miền Nam"}), "utf-8")
    vs = T.list_voices()
    assert vs["ha_my"].region == "nam" and vs["ha_my"].wav == d / "prompt.wav"
    main(["voices"])
    assert "ha_my" in capsys.readouterr().out


def _find(q: str, vs=None) -> T.Voice:
    v = T.find_voice(q, vs)
    assert v is not None, q
    return v


def _voice(name: str, label: str) -> T.Voice:
    from pathlib import Path

    return T.Voice(name, Path("x.wav"), "Chào.", label=label)


def test_find_voice_by_id_label_and_without_diacritics():
    vs = {"thanh_thanh": _voice("thanh_thanh", "Thanh Thanh"), "thanh": _voice("thanh", "Thành")}
    assert _find("thanh", vs).label == "Thành"  # mã thư mục
    assert _find("Thành", vs).name == "thanh"
    assert _find("thanh thanh", vs).name == "thanh_thanh"
    assert _find("THANH-THANH", vs).name == "thanh_thanh"
    assert _find("Thánh", vs).name == "thanh"  # sai dấu: chỉ một giọng khớp khi bỏ dấu
    assert T.find_voice("lan", vs) is None


def test_check_reports_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PARROT_TTS_HOME", str(tmp_path))
    with pytest.raises(SystemExit):
        main(["check"])
    out = capsys.readouterr().out
    assert "THIẾU llm Parrot-TTS" in out and "parrot-tts download" in out


def test_save_marks_ai(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    f = tmp_path / "a.wav"
    T.save(f, np.zeros(2400, dtype="float32"), 24000)
    with sf.SoundFile(str(f)) as s:
        assert s.comment == T.AI_MARK and s.samplerate == 24000


def test_server_api_with_fake_model(tmp_path):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from pathlib import Path

    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from parrot_tts.server import make_handler
    from parrot_tts.usage import Usage

    class FakeVV:
        def speak(self, text, voice=None, dialect=None):
            self.last = (text, voice, dialect)
            return np.zeros(2400, dtype="float32"), 24000

    vv = FakeVV()
    voices = {"thanh": T.Voice("thanh", Path("x.wav"), "Chào.", region="nam", label="Thành", aliases=("san",))}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(vv, voices, Usage(None), "Parrot-TTS (llm.pt)"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        got = json.loads(urllib.request.urlopen(base + "/voices").read())
        assert got["voices"] == [{"id": "thanh", "label": "Thành", "source": "Parrot-TTS", "aliases": ["san"]}]

        def post(body):
            req = urllib.request.Request(base + "/tts", json.dumps(body).encode(), {"Content-Type": "application/json"})
            return urllib.request.urlopen(req)

        r = post({"text": "Giảm  50%.", "voice": "thanh"})
        assert r.headers["Content-Type"] == "audio/wav" and r.headers["Access-Control-Allow-Origin"] == "*"
        assert json.loads(r.headers["X-Normalized-Text"]) == "giảm năm mươi phần trăm."
        f = tmp_path / "o.wav"
        f.write_bytes(r.read())
        with sf.SoundFile(str(f)) as s:
            assert s.comment == T.AI_MARK and s.samplerate == 24000
        assert vv.last == ("Giảm 50%.", "thanh", "nam")  # dialect theo giọng khi trang không gửi
        post({"text": "Chào.", "voice": "san"}).read()  # mã cũ vẫn đọc được, đếm vào mã chính
        assert vv.last[1] == "thanh"
        with pytest.raises(urllib.error.HTTPError) as e:
            post({"text": "Chào.", "voice": "lan"})
        assert e.value.code == 400
        assert json.loads(urllib.request.urlopen(base + "/stats").read())["voices"]["thanh"]["total"] == 2
    finally:
        srv.shutdown()
