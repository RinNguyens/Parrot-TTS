"""Parrot-TTS: đọc tiếng Việt bằng giọng mẫu, chạy trên máy của bạn.

parrot-tts download                                  # tải model và mã CosyVoice về ~/.cache/parrot-tts
parrot-tts check                                     # máy đã đủ thứ để đọc chưa
parrot-tts voices                                    # các giọng có sẵn
parrot-tts tts "Xin chào các bạn." --voice "Thanh Thanh" -o chao.wav
parrot-tts serve                                     # máy đọc cho trang web, http://localhost:8000
parrot-tts tts "Xin chào." --prompt-wav mau.wav --prompt-text "lời đúng trong mau.wav" -o chao.wav
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
COSYVOICE_GIT = "https://github.com/FunAudioLLM/CosyVoice.git"
COSYVOICE_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"  # cùng commit lúc train và chấm
LLM_REPO = os.environ.get("PARROT_TTS_REPO", "rin98s/Parrot-TTS")
LLM_FILE = "llm.pt"
# File của base model mà lúc đọc không cần: llm gốc (đã thay bằng llm Parrot-TTS), bản RL, ONNX cho TensorRT,
# bộ tách token theo lô (chỉ dùng khi train), ảnh minh hoạ.
# CosyVoice-BlankEN/model.safetensors: trọng số Qwen gốc, bị llm.pt ghi đè (tts._qwen_from_config).
BASE_SKIP = [
    "llm.pt",
    "llm.rl.pt",
    "flow.decoder.estimator.*",
    "speech_tokenizer_v3.batch.onnx",
    "asset/*",
    "CosyVoice-BlankEN/model.safetensors",
]
BASE_NEED = ["cosyvoice3.yaml", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN"]


def cmd_download(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download  # pyright: ignore[reportMissingImports]  # gói tuỳ chọn [tts]

    from parrot_tts.tts import home

    h = home()
    h.mkdir(parents=True, exist_ok=True)
    cosy = h / "CosyVoice"
    if not (cosy / ".git").exists():
        subprocess.run(["git", "clone", "--recursive", COSYVOICE_GIT, str(cosy)], check=True)
    subprocess.run(["git", "-C", str(cosy), "checkout", "--quiet", COSYVOICE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(cosy), "submodule", "update", "--init", "--recursive"], check=True)
    print("tải base model", BASE_REPO)
    snapshot_download(BASE_REPO, local_dir=str(h / "Fun-CosyVoice3-0.5B"), ignore_patterns=BASE_SKIP)
    print("tải llm Parrot-TTS", args.repo)
    src = hf_hub_download(args.repo, LLM_FILE, local_dir=str(h / "hf"))
    dst = h / "llm.pt"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(Path(src).resolve())
    try:
        voices = snapshot_download(args.repo, allow_patterns=["voices/*"], local_dir=str(h))
        print("giọng:", sorted(p.name for p in (Path(voices) / "voices").iterdir() if p.is_dir()))
    except Exception as e:  # repo chưa có giọng
        print("không tải được giọng:", str(e)[:120])
    print(f"Xong: {h}. Kiểm tra: parrot-tts check")


def cmd_check(_: argparse.Namespace) -> None:
    """In từng thứ máy cần để đọc; thoát mã 1 nếu thiếu."""
    import importlib.util

    from parrot_tts.tts import home, list_voices

    h = home()
    base = h / "Fun-CosyVoice3-0.5B"
    rows = [(f"base model: {f}", (base / f).exists()) for f in BASE_NEED]
    rows.append(("llm Parrot-TTS: llm.pt", (h / "llm.pt").exists()))
    rows.append(("mã CosyVoice", (h / "CosyVoice" / "cosyvoice").is_dir()))
    rows.append(("Matcha-TTS (submodule)", (h / "CosyVoice" / "third_party" / "Matcha-TTS" / "matcha").is_dir()))
    for mod in ("torch", "torchaudio", "onnxruntime", "hyperpyyaml", "whisper", "transformers"):
        rows.append((f"thư viện: {mod}", importlib.util.find_spec(mod) is not None))
    rows.append((f"giọng: {len(list_voices())}", bool(list_voices())))
    for what, ok in rows:
        print(("  có   " if ok else "  THIẾU ") + what)
    if importlib.util.find_spec("torch"):
        import torch  # pyright: ignore[reportMissingImports]  # gói tuỳ chọn [tts]

        print("  chạy trên:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (chậm)")
    if not all(ok for _, ok in rows):
        print(f"\nThiếu file model/giọng: parrot-tts download. Thiếu thư viện: pip install 'parrot-tts[tts]'. ({h})")
        raise SystemExit(1)
    print('Đủ. Thử: parrot-tts tts "Xin chào các bạn." --voice <tên> -o chao.wav')


def cmd_voices(_: argparse.Namespace) -> None:
    from parrot_tts.tts import list_voices

    vs = list_voices()
    if not vs:
        print("chưa có giọng nào: chạy `parrot-tts download`, hoặc dùng --prompt-wav/--prompt-text")
    for v in vs.values():
        print(f"{v.label:14} {v.name:14} {v.gender:4} {v.region:4} {v.desc}")


def cmd_tts(args: argparse.Namespace) -> None:
    from parrot_tts.tts import ParrotTTS, save

    text = args.text if args.text != "-" else sys.stdin.read()
    vv = ParrotTTS(base=args.base, llm=args.llm, asr="" if args.no_clean else None)
    audio, sr = vv.speak(
        text,
        voice=args.voice,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        dialect=args.dialect,
        seed=args.seed,
        clean=not args.no_clean,
    )
    save(args.out, audio, sr)
    print(f"{args.out}: {len(audio) / sr:.1f} s")


def cmd_clean(args: argparse.Namespace) -> None:
    import soundfile as sf

    from parrot_tts.tts import ParrotTTS

    vv = ParrotTTS.__new__(ParrotTTS)  # chỉ cần ASR, không nạp model TTS
    vv._asr_name, vv._asr = "openai/whisper-large-v3-turbo", None
    a, sr, text, note = vv.clean_prompt(args.wav, args.text)
    sf.write(args.out, a, sr)
    print(f"{note}\nvăn bản: {text}")


def cmd_serve(args: argparse.Namespace) -> None:
    from parrot_tts.server import serve
    from parrot_tts.tts import home

    serve(args.host, args.port, args.usage or home() / "usage.json", base=args.base, llm=args.llm)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="parrot-tts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("download", help="tải model và mã CosyVoice về $PARROT_TTS_HOME")
    s.add_argument("--repo", default=LLM_REPO)
    s.set_defaults(fn=cmd_download)
    s = sub.add_parser("check", help="kiểm tra model, mã CosyVoice, thư viện, giọng")
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("voices", help="liệt kê giọng có sẵn")
    s.set_defaults(fn=cmd_voices)
    s = sub.add_parser("tts", help="đọc văn bản ra file wav")
    s.add_argument("text", help='văn bản, hoặc "-" để đọc từ stdin')
    s.add_argument("-o", "--out", default="out.wav")
    s.add_argument("--voice", help="tên hoặc mã giọng (xem `parrot-tts voices`)")
    s.add_argument("--prompt-wav", help="câu mẫu 5–15 s của giọng muốn dùng")
    s.add_argument("--prompt-text", help="lời đúng trong câu mẫu")
    s.add_argument("--dialect", choices=["bac", "nam"], help="cách đọc số/viết tắt (mặc định theo giọng)")
    s.add_argument("--seed", type=int)
    s.add_argument("--no-clean", action="store_true", help="không làm sạch câu mẫu (không cần Whisper)")
    s.add_argument("--base", help="thư mục Fun-CosyVoice3-0.5B (mặc định trong $PARROT_TTS_HOME)")
    s.add_argument("--llm", help="llm.pt (mặc định trong $PARROT_TTS_HOME)")
    s.set_defaults(fn=cmd_tts)
    s = sub.add_parser("serve", help="máy đọc cho trang web Parrot Voice (cùng API với tts_server.py)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--usage", type=Path, help="file đếm lượt đọc (mặc định $PARROT_TTS_HOME/usage.json)")
    s.add_argument("--base")
    s.add_argument("--llm")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("clean-prompt", help="làm sạch một câu mẫu (cắt chữ đầu cho khớp văn bản)")
    s.add_argument("wav")
    s.add_argument("text")
    s.add_argument("-o", "--out", default="prompt_clean.wav")
    s.set_defaults(fn=cmd_clean)
    args = p.parse_args(argv)
    args.fn(args)
