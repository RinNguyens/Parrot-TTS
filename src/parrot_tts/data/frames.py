"""Chia khung và đổi mức ra dB, dùng chung cho QA phiên thu và VAD năng lượng."""

from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray


def frame_power(mono: NDArray[np.float64], frame: int) -> NDArray[np.float64]:
    """Công suất trung bình (RMS²) của từng khung đủ ``frame`` mẫu, không chồng lấn.

    Khung cuối thiếu mẫu bị bỏ: một khung vài mẫu có công suất rất lệch, kéo sai phân vị nền ồn.
    """
    if mono.ndim != 1:
        raise ValueError(f"cần tín hiệu một chiều, gặp mảng {mono.ndim} chiều")
    if frame < 1:
        raise ValueError(f"độ dài khung phải >= 1 mẫu, gặp {frame}")
    n_frames = mono.size // frame
    return (mono[: n_frames * frame] ** 2).reshape(n_frames, frame).mean(axis=1)


def run_lengths(mask: NDArray[np.bool_]) -> list[int]:
    """Độ dài các đoạn ``True`` liên tiếp."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist()


def power_db(power: NDArray[np.float64] | float, floor_db: float) -> NDArray[np.float64]:
    """10·log10(công suất), chặn dưới ở ``floor_db`` để lặng tuyệt đối không ra -inf."""
    return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), 10.0 ** (floor_db / 10.0)))


def read_float(path: Path) -> tuple[NDArray[np.float64], int]:
    """Đọc file âm thanh ra float64 trong [-1, 1] (một chiều nếu mono)."""
    data, rate = sf.read(str(path), dtype="float64")
    return np.asarray(data, dtype=np.float64), int(rate)
