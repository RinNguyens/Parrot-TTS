"""Audio tổng hợp cho test data pipeline: sóng sin, ồn trắng, lặng, tín hiệu bị clip.

Không commit audio thật; mọi file được tạo trong ``tmp_path`` bằng numpy + soundfile.
Mức tính theo dBFS với full scale = 1.0 (RMS của sin full scale là -3,01 dBFS).
"""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

SR = 48_000

Audio = NDArray[np.float64]


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def sine(duration_s: float, rms_dbfs: float, freq_hz: float = 1000.0, sr: int = SR) -> Audio:
    """Sin có RMS đúng ``rms_dbfs`` (biên độ = RMS * căn 2)."""
    t = np.arange(round(duration_s * sr)) / sr
    return db_to_amp(rms_dbfs) * np.sqrt(2.0) * np.sin(2.0 * np.pi * freq_hz * t)


def noise(duration_s: float, rms_dbfs: float, seed: int = 0, sr: int = SR) -> Audio:
    """Ồn trắng Gauss có RMS ``rms_dbfs``."""
    rng = np.random.default_rng(seed)
    return db_to_amp(rms_dbfs) * rng.standard_normal(round(duration_s * sr))


def silence(duration_s: float, sr: int = SR) -> Audio:
    return np.zeros(round(duration_s * sr))


def speech_like(
    segments: list[tuple[float, float]],
    total_s: float,
    speech_dbfs: float = -20.0,
    noise_dbfs: float = -70.0,
    sr: int = SR,
    seed: int = 0,
) -> Audio:
    """Ồn nền suốt file + sin ở các đoạn ``(start_s, end_s)`` (giả làm tiếng nói)."""
    data = noise(total_s, noise_dbfs, seed=seed, sr=sr)
    for start_s, end_s in segments:
        start, end = round(start_s * sr), round(end_s * sr)
        data[start:end] += sine((end - start) / sr, speech_dbfs, sr=sr)
    return data


def write_wav(path: Path, data: Audio, sr: int = SR, subtype: str = "PCM_24") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sr, subtype=subtype, format="WAV")
    return path


def good_take(path: Path, seed: int = 0) -> Path:
    """File thu đạt QA: nền ồn -70 dBFS, tiếng -20 dBFS, đỉnh khoảng -17 dBFS."""
    data = speech_like([(0.4, 1.6), (2.0, 3.2)], total_s=3.6, seed=seed)
    return write_wav(path, data)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
