"""Làm sạch câu mẫu giọng: cắt bỏ chữ đầu để câu mẫu bắt đầu đúng ở ranh giới chữ và văn bản khớp audio từ chữ đầu.

Model P đọc rỗng khi chữ đầu của văn bản câu mẫu không có trong audio (đoạn cắt giữa chữ). Luật tự kiểm chứng:
thử các điểm lặng trong 1,5 s đầu, nhận điểm mà ASR nghe ra đúng văn bản từ chữ k (1 ≤ k ≤ 4), chọn điểm lặng nhất.
Chỉ dùng numpy; ASR do người gọi đưa vào (hàm transcribe).
"""

from __future__ import annotations

import re

FRAME_S = 0.02
MIN_CUT_S = 0.05
MAX_CUT_S = 1.5
MAX_DROP = 4
HEAD_N = 3
LEAD_SIL_S = 0.1
FADE_S = 0.01

_PUNCT = re.compile(r"[^\w]+", re.UNICODE)


def norm(w: str) -> str:
    return _PUNCT.sub("", w.lower())


def nwords(s: str) -> list[str]:
    return [norm(w) for w in s.split() if norm(w)]


def candidates(frame_db: list[float], frame_s: float = FRAME_S) -> list[float]:
    """Giây của các khung cực tiểu năng lượng cục bộ trong [MIN_CUT_S, MAX_CUT_S], theo thời gian."""
    out = []
    for i in range(1, len(frame_db) - 1):
        t = i * frame_s
        if t < MIN_CUT_S - 1e-9 or t > MAX_CUT_S + 1e-9:
            continue
        if frame_db[i] <= frame_db[i - 1] and frame_db[i] < frame_db[i + 1]:
            out.append(round(t, 4))
    return out


def match_k(asr_text: str, text: str) -> int | None:
    """k (1..MAX_DROP) nếu HEAD_N chữ đầu Whisper nghe = HEAD_N chữ văn bản từ chữ k; None nếu không."""
    a = nwords(asr_text)[:HEAD_N]
    t = text.split()
    if len(a) < HEAD_N:
        return None
    for k in range(1, MAX_DROP + 1):
        if nwords(" ".join(t[k:]))[:HEAD_N] == a:
            return k
    return None


def pick_cut(text: str, cands: list[float], transcribe, energy: dict[float, float] | None = None):
    """-> (giây cắt, văn bản mới, k, lời Whisper) hoặc None. transcribe(giây cắt) -> str; energy: giây cắt -> dB."""
    energy = energy or {}
    ok: list[tuple[float, str]] = []
    k0 = None
    for c in cands:
        heard = transcribe(c)
        k = match_k(heard, text)
        if k0 is None:
            if k is not None:
                k0, ok = k, [(c, heard)]
            continue
        if k != k0:
            break
        ok.append((c, heard))
    if k0 is None:
        return None
    c, heard = min(ok, key=lambda x: (energy.get(x[0], 0.0), x[0]))
    new = " ".join(text.split()[k0:])
    return c, new[:1].upper() + new[1:], k0, heard


def frames_db(x, sr: int, frame_s: float = FRAME_S) -> list[float]:
    import numpy as np

    f = int(sr * frame_s)
    n = len(x) // f
    if n == 0:
        return []
    e = (np.asarray(x[: n * f], dtype=float).reshape(n, f) ** 2).mean(1)
    return (10 * np.log10(e + 1e-12)).tolist()


def apply_cut(x, sr: int, cut_s: float):
    import numpy as np

    y = np.asarray(x[int(cut_s * sr) :], dtype=float).copy()
    nf = min(len(y), int(FADE_S * sr))
    y[:nf] *= np.linspace(0, 1, nf)
    return np.concatenate([np.zeros(int(LEAD_SIL_S * sr)), y])
