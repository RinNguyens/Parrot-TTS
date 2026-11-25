"""Máy đọc chạy trên máy của bạn cho trang web Parrot Voice (`vietvoice serve`).

Cùng API với infra/vast/tts_server.py, nên trang /create-tts và /giong dùng được ngay (ô "Máy đọc" để
http://localhost:8000):

- GET  /voices  các giọng đã cài (`vietvoice voices`), mã = tên thư mục (thanh_thanh, thanh...), kèm mã cũ (aliases)
                mà POST /tts cũng nhận (fr07 = Trúc, fleurs = Hải...).
- POST /tts     JSON {"text": "...", "voice": "thanh_thanh", "dialect": "nam"} -> audio/wav 24 kHz mono, có nhãn AI.
- GET  /stats   lượt đọc theo giọng (chỉ đếm, không lưu văn bản), cho bảng xếp hạng /giong.

Chỉ nghe trên 127.0.0.1. Đọc lần lượt từng yêu cầu (một model). Trên CPU mỗi giây audio mất khoảng 4–5 giây.
"""

from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_CHARS = 3000


def wav_bytes(audio, sr: int) -> bytes:
    import soundfile as sf

    from vietvoice.tts import AI_MARK

    buf = io.BytesIO()
    with sf.SoundFile(buf, "w", sr, 1, "PCM_16", format="WAV") as f:
        f.comment = AI_MARK
        f.software = "VietVoice"
        f.write(audio)
    return buf.getvalue()


def make_handler(vv, voices: dict, usage, llm_name: str):
    """vv: VietVoice (hoặc vật giả có .speak), voices: mã -> Voice, usage: vietvoice.usage.Usage."""
    from vietvoice.tts import tts_text

    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            if path == "/stats":
                return self._json(200, usage.stats())
            if path == "/voices":
                vs = [
                    {"id": k, "label": v.label, "source": v.credit or "VietVoice", "aliases": list(v.aliases)}
                    for k, v in voices.items()
                ]
                return self._json(200, {"llm": llm_name, "voices": vs})
            self._json(404, {"error": "không có đường dẫn này"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/tts":
                return self._json(404, {"error": "không có đường dẫn này"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "JSON không đọc được"})
            text = " ".join(str(req.get("text", "")).split())
            vid = str(req.get("voice", ""))
            voice = voices.get(vid) or next((v for v in voices.values() if vid in v.aliases), None)
            if voice is not None:
                vid = voice.name  # đếm lượt và đọc theo mã chính
            if not text:
                return self._json(400, {"error": "Chưa có văn bản"})
            if len(text) > MAX_CHARS:
                return self._json(400, {"error": f"Văn bản dài quá {MAX_CHARS} ký tự"})
            if voice is None:
                return self._json(400, {"error": "Giọng này chưa có trên máy đọc"})
            dialect = req.get("dialect") if req.get("dialect") in ("bac", "nam") else voice.region
            t0 = time.time()
            try:
                with lock:
                    audio, sr = vv.speak(text, voice=vid, dialect=dialect)
            except Exception as e:  # lỗi model: báo cho trang, máy đọc vẫn chạy
                return self._json(500, {"error": f"đọc lỗi: {e}"[:300]})
            body = wav_bytes(audio, sr)
            usage.add(vid)
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Normalized-Text", json.dumps(tts_text(text, dialect), ensure_ascii=True))
            self.send_header("X-Seconds", f"{time.time() - t0:.2f}")
            self.send_header("Access-Control-Expose-Headers", "X-Normalized-Text, X-Seconds")
            self.end_headers()
            self.wfile.write(body)
            print(f"[{voice.label}] {time.time() - t0:.1f}s: {text[:60]}", flush=True)

        def log_message(self, format: str, *args) -> None:  # noqa: A002 (tên của lớp cha)
            pass

    return Handler


def serve(host: str, port: int, usage_path: Path | None, base=None, llm=None) -> None:
    from vietvoice.tts import VietVoice, home, list_voices
    from vietvoice.usage import Usage

    voices = list_voices()
    if not voices:
        raise SystemExit("chưa có giọng nào: chạy `vietvoice download` trước")
    vv = VietVoice(base=base, llm=llm, asr="")
    llm_path = Path(llm) if llm else home() / "llm.pt"
    handler = make_handler(vv, voices, Usage(usage_path), f"VietVoice ({llm_path.resolve().name})")
    names = ", ".join(v.label for v in voices.values())
    print(f"Máy đọc VietVoice: {names}. Nghe ở http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), handler).serve_forever()
