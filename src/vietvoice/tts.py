"""Đọc văn bản tiếng Việt bằng giọng mẫu, chạy trên máy người dùng (GPU NVIDIA; CPU chạy được nhưng chậm).

Model = Fun-CosyVoice3-0.5B (Apache-2.0) với llm.pt của VietVoice thay cho llm gốc; flow, hift giữ nguyên.
Cách đọc giống hệt lúc chấm điểm (infra/vast/eval_tts.py, chế độ --guard):
  - chuẩn hoá văn bản (số, ngày, viết tắt -> chữ) bằng vietvoice.text.normalize;
  - tách đoạn ≤ 30 âm tiết theo câu và dấu phẩy;
  - mỗi đoạn đọc tối đa `tries` lần, lấy lần có độ dài hợp lý (0,5–2 lần độ dài dự kiến);
  - câu mẫu được làm sạch trước (vietvoice.prompt) nếu có ASR.

    from vietvoice.tts import VietVoice
    vv = VietVoice()                      # tìm model ở $VIETVOICE_HOME (mặc định ~/.cache/vietvoice)
    audio, sr = vv.speak("Xin chào các bạn.", voice="Thanh Thanh")   # tên hiển thị hoặc mã thư mục (thanh_thanh)
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
SEC_PER_SYL = 0.505  # trung vị tốc độ đọc đo trên model FLEURS
RATIO_OK = (0.5, 2.0)
MAX_SYL = 30
GAP_S = 0.15
AI_MARK = "Tạo bởi VietVoice (giọng tổng hợp bằng AI)"


def home() -> Path:
    return Path(os.environ.get("VIETVOICE_HOME", Path.home() / ".cache" / "vietvoice"))


@dataclass
class Voice:
    name: str
    wav: Path
    text: str
    region: str = "bac"
    desc: str = ""
    label: str = ""  # tên hiển thị, ví dụ "Thanh Thanh"; name là mã thư mục (thanh_thanh)
    gender: str = ""
    credit: str = ""
    aliases: tuple[str, ...] = ()  # mã cũ của cùng giọng (fr07, fleurs...), để trang web và script cũ vẫn gọi được


def voices_dirs() -> list[Path]:
    """Thư mục giọng: $VIETVOICE_HOME/voices, rồi voices/ đi kèm gói (nếu có)."""
    return [d for d in (home() / "voices", Path(__file__).resolve().parent / "voices") if d.is_dir()]


def list_voices() -> dict[str, Voice]:
    out: dict[str, Voice] = {}
    for d in voices_dirs():
        for meta in sorted(d.glob("*/voice.json")):
            v = json.loads(meta.read_text(encoding="utf-8"))
            name = meta.parent.name
            out.setdefault(
                name,
                Voice(
                    name,
                    meta.parent / v.get("wav", "prompt.wav"),
                    v["text"],
                    v.get("region", "bac"),
                    v.get("desc", ""),
                    v.get("name", name),
                    v.get("gender", ""),
                    v.get("credit", ""),
                    tuple(v.get("aliases", ())),
                ),
            )
    return out


def fold(s: str) -> str:
    """Bỏ dấu, hoa/thường, khoảng trắng và gạch: "Thanh Thanh", "thanh_thanh", "THANH-THANH" -> "thanhthanh"."""
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if c.isalnum() and not unicodedata.combining(c))


def find_voice(name: str, voices: dict[str, Voice] | None = None) -> Voice | None:
    """Tìm giọng theo mã thư mục hoặc tên hiển thị. Ưu tiên khớp đúng từng chữ (Thành khác Thanh), rồi mới bỏ dấu."""
    voices = list_voices() if voices is None else voices
    if name in voices:
        return voices[name]
    for v in voices.values():
        if name in v.aliases:
            return v
    exact = [v for v in voices.values() if v.label.lower() == name.strip().lower()]
    if exact:
        return exact[0]
    loose = [v for v in voices.values() if fold(name) in (fold(v.name), fold(v.label))]
    return loose[0] if len(loose) == 1 else None


def tts_text(text: str, dialect: str) -> str:
    """Văn bản đưa vào model, như cột `tts` của bộ chấm: giữ nguyên câu nếu chuẩn hoá chỉ đổi hoa/thường,
    ngược lại dùng bản chuẩn hoá (số, ngày, viết tắt đã thành chữ)."""
    from vietvoice.text.normalize import normalize

    text = " ".join(text.split())
    n = normalize(text, dialect if dialect in ("bac", "nam") else "bac")
    return text if n == text.lower() else n


def split_text(text: str, max_syl: int = MAX_SYL) -> list[str]:
    """Tách theo câu (. ! ? ; :), câu quá dài tách tiếp theo dấu phẩy, rồi gộp mẩu ngắn cạnh nhau."""
    pieces = []
    for sent in re.split(r"(?<=[.!?;:])\s+", text.strip()):
        if len(sent.split()) <= max_syl:
            pieces.append(sent)
        else:
            pieces += [p for p in re.split(r"(?<=,)\s+", sent) if p]
    chunks: list[str] = []
    for p in pieces:
        if chunks and len(chunks[-1].split()) + len(p.split()) <= max_syl:
            chunks[-1] += " " + p
        else:
            chunks.append(p)
    return chunks or [text]


def length_ok(n_samples: int, sr: int, chunk: str) -> tuple[bool, float]:
    expect = max(len(chunk.split()), 1) * SEC_PER_SYL
    ratio = n_samples / sr / expect
    return RATIO_OK[0] <= ratio <= RATIO_OK[1], ratio


def _stub_wetext() -> None:
    """CosyVoice nạp bộ chuẩn hoá wetext (Trung/Anh) tải từ modelscope.cn, hay bị chặn. Ta tự chuẩn hoá tiếng Việt."""

    class _NoWetext:
        def __init__(self, *a, **k):
            raise RuntimeError("VietVoice không dùng wetext")

    sys.modules.setdefault("wetext", types.SimpleNamespace(Normalizer=_NoWetext))  # type: ignore[arg-type]


def _cosyvoice_on_path() -> None:
    root = Path(os.environ.get("COSYVOICE_ROOT", home() / "CosyVoice"))
    for p in (root, root / "third_party" / "Matcha-TTS"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _qwen_from_config(base: Path) -> None:
    """CosyVoice dựng llm bằng Qwen2ForCausalLM.from_pretrained(CosyVoice-BlankEN), cần model.safetensors (~1 GB),
    rồi llm.pt ghi đè toàn bộ trọng số đó. Không có file đó thì dựng Qwen từ config.json (trọng số ngẫu nhiên, bị
    llm.pt thay ngay), nên người dùng khỏi tải 1 GB vô ích."""
    if (base / "CosyVoice-BlankEN" / "model.safetensors").exists():
        return
    import cosyvoice.llm.llm as L  # pyright: ignore[reportMissingImports]  # gói tuỳ chọn [tts]
    from transformers import AutoConfig  # pyright: ignore[reportMissingImports]  # gói tuỳ chọn [tts]

    orig = L.Qwen2ForCausalLM

    class _FromConfig(orig):  # type: ignore[misc,valid-type]
        @classmethod
        def from_pretrained(cls, path, *a, **k):
            return orig(AutoConfig.from_pretrained(path))

    L.Qwen2ForCausalLM = _FromConfig


def model_dir(base: Path, llm: Path | None) -> Path:
    """Thư mục model = base model, llm.pt trỏ sang llm của VietVoice."""
    if llm is None:
        return base
    d = home() / "model"
    d.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        link = d / f.name
        if f.name != "llm.pt" and not (link.exists() or link.is_symlink()):
            link.symlink_to(f.resolve())
    link = d / "llm.pt"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(llm.resolve())
    return d


class VietVoice:
    def __init__(self, base: str | Path | None = None, llm: str | Path | None = None, asr: str | None = None):
        """base: thư mục Fun-CosyVoice3-0.5B; llm: llm.pt của VietVoice. Mặc định lấy trong $VIETVOICE_HOME.
        asr: model Whisper để làm sạch câu mẫu ("" = không làm sạch); mặc định whisper-large-v3-turbo nếu có
        transformers."""
        base = Path(base) if base else home() / "Fun-CosyVoice3-0.5B"
        llm = Path(llm) if llm else home() / "llm.pt"
        if not (base / "cosyvoice3.yaml").exists():
            raise FileNotFoundError(f"không thấy base model ở {base}: chạy `vietvoice download` trước")
        if not llm.exists():
            raise FileNotFoundError(f"không thấy {llm}: chạy `vietvoice download` trước")
        _stub_wetext()
        _cosyvoice_on_path()
        _qwen_from_config(base)
        from cosyvoice.cli.cosyvoice import AutoModel  # pyright: ignore[reportMissingImports]  # gói tuỳ chọn [tts]

        self.model = AutoModel(model_dir=str(model_dir(base, llm)))
        self.sr = self.model.sample_rate
        self._asr_name = "openai/whisper-large-v3-turbo" if asr is None else asr
        self._asr = None

    # ---------- câu mẫu ----------

    def _transcriber(self):
        if not self._asr_name:
            return None
        if self._asr is None:
            try:
                import torch  # pyright: ignore[reportMissingImports]  # gói tuỳ chọn [tts]
                from transformers import pipeline  # pyright: ignore[reportMissingImports]  # gói tuỳ chọn [tts]
            except ImportError:
                self._asr_name = ""
                return None
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            dt = torch.float16 if dev != "cpu" else torch.float32
            self._asr = pipeline("automatic-speech-recognition", model=self._asr_name, torch_dtype=dt, device=dev)
        return self._asr

    def clean_prompt(self, wav: str | Path, text: str) -> tuple[np.ndarray, int, str, str]:
        """-> (audio, sr, văn bản, ghi chú). Không có ASR hoặc không tìm được điểm cắt: giữ nguyên."""
        import soundfile as sf

        from vietvoice.prompt import apply_cut, candidates, frames_db, pick_cut

        a, sr = sf.read(str(wav), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        asr = self._transcriber()
        if asr is None:
            return a, sr, text, "giữ nguyên (không có ASR)"
        kw = {"language": "vi", "task": "transcribe"}
        db = frames_db(a, sr)
        cands = candidates(db)
        energy = {c: db[round(c / 0.02)] for c in cands}

        def transcribe(c: float) -> str:
            y = apply_cut(a, sr, c).astype("float32")
            return asr({"raw": y, "sampling_rate": sr}, generate_kwargs=kw)["text"]

        res = pick_cut(text, cands, transcribe, energy)
        if res is None:
            return a, sr, text, "giữ nguyên (không tìm được điểm cắt khớp văn bản)"
        cut, new_text, k, _ = res
        return apply_cut(a, sr, cut).astype("float32"), sr, new_text, f"cắt {cut:.2f} s, bỏ {k} chữ đầu"

    # ---------- đọc ----------

    def speak(
        self,
        text: str,
        voice: str | None = None,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
        dialect: str | None = None,
        tries: int = 3,
        seed: int | None = None,
        clean: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Đọc `text` bằng giọng có tên (`voice`) hoặc câu mẫu riêng (prompt_wav + prompt_text đúng lời trong audio)."""
        import tempfile

        import soundfile as sf
        import torch  # pyright: ignore[reportMissingImports]  # gói tuỳ chọn [tts]

        if voice:
            v = find_voice(voice)
            if v is None:
                raise KeyError(f"không có giọng {voice!r}; xem `vietvoice voices`")
            prompt_wav, prompt_text, dialect = v.wav, v.text, dialect or v.region
            clean = False  # giọng đóng gói sẵn đã được làm sạch
        if not prompt_wav or not prompt_text:
            raise ValueError("cần voice, hoặc prompt_wav + prompt_text")
        dialect = dialect or "bac"
        if clean:
            a, sr, prompt_text, _ = self.clean_prompt(prompt_wav, prompt_text)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, a, sr)
            prompt_wav = tmp.name
        if seed is not None:
            from cosyvoice.utils.common import set_all_random_seed  # pyright: ignore[reportMissingImports]  # gói tuỳ chọn [tts]

            set_all_random_seed(seed)
        gap = np.zeros(int(GAP_S * self.sr), dtype="float32")
        out: list[np.ndarray] = []
        for chunk in split_text(tts_text(text, dialect)):
            best, best_key = None, float("inf")
            for _ in range(tries):
                try:
                    parts = [
                        o["tts_speech"]
                        for o in self.model.inference_zero_shot(
                            chunk, INSTRUCT + prompt_text, str(prompt_wav), stream=False, text_frontend=False
                        )
                    ]
                except RuntimeError:  # llm sinh quá ít token: vocoder lỗi, đọc lại
                    parts = []
                a = torch.cat(parts, dim=1).squeeze(0).cpu().numpy() if parts else np.zeros(1, dtype="float32")
                ok, ratio = length_ok(len(a), self.sr, chunk)
                key = 0.0 if ok else 1 + abs(np.log(max(ratio, 1e-3)))
                if key < best_key:
                    best, best_key = a, key
                if ok:
                    break
            out += [best, gap]
        return np.concatenate(out[:-1]), self.sr


def save(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Ghi WAV 16-bit kèm nhãn "do AI tạo" trong metadata (Luật AI 134/2025 yêu cầu đánh dấu đọc được bằng máy)."""
    import soundfile as sf

    with sf.SoundFile(str(path), "w", sr, 1, "PCM_16") as f:
        f.comment = AI_MARK
        f.software = "VietVoice"
        f.write(audio)
