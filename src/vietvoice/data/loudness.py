"""Bước 2 pipeline: đo loudness tích hợp ITU-R BS.1770 (pyloudnorm) và đưa về ``target_lufs``.

Chỉ nhân một hệ số gain, không nén, không limiter. Nếu gain cần thiết đẩy đỉnh (sample peak)
vượt ``qa.max_peak_dbfs`` thì ném ``PeakLimitError`` thay vì để clip: câu đó cần xem lại.
Ghi ra WAV 24-bit, cùng tần số lấy mẫu. Thông số trong ``loudness.yaml`` và ``qa.yaml``.

Dòng lệnh: ``python -m vietvoice.data.loudness <vào.wav> <ra.wav>``.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
from numpy.typing import NDArray
from pydantic import Field

from vietvoice.data.config import FrozenModel, load_qa_config, load_yaml_model
from vietvoice.data.frames import read_float


class LoudnessConfig(FrozenModel):
    target_lufs: float
    tolerance_lu: float = Field(gt=0)
    max_passes: int = Field(ge=1)
    output_subtype: str


class PeakLimitError(ValueError):
    """Chuẩn hoá sẽ đẩy đỉnh vượt giới hạn QA."""


@dataclass(frozen=True)
class LoudnessResult:
    input_lufs: float
    output_lufs: float
    gain_db: float
    peak_dbfs: float


def load_loudness_config(path: Path | None = None) -> LoudnessConfig:
    return load_yaml_model(LoudnessConfig, "loudness", path)


def measure_loudness(audio: NDArray[np.float64], sample_rate: int) -> float:
    """Loudness tích hợp (LUFS); tín hiệu ngắn hơn một khối đo 400 ms thì ném ``ValueError``."""
    meter = pyloudnorm.Meter(sample_rate)
    if audio.shape[0] < round(meter.block_size * sample_rate):
        raise ValueError(
            f"tín hiệu quá ngắn để đo loudness: {audio.shape[0] / sample_rate:.3f} s < {meter.block_size} s"
        )
    return float(meter.integrated_loudness(audio))


def normalize(
    audio: NDArray[np.float64],
    sample_rate: int,
    config: LoudnessConfig | None = None,
    max_peak_dbfs: float | None = None,
) -> tuple[NDArray[np.float64], LoudnessResult]:
    """Nhân gain để loudness về đích; trả (tín hiệu mới, kết quả đo)."""
    cfg = config or load_loudness_config()
    peak_limit = load_qa_config().qa.max_peak_dbfs if max_peak_dbfs is None else max_peak_dbfs
    input_lufs = measure_loudness(audio, sample_rate)
    if not np.isfinite(input_lufs):
        raise ValueError("không đo được loudness (tín hiệu lặng hoặc dưới cổng -70 LUFS), không chuẩn hoá được")

    gain_db = 0.0
    current = input_lufs
    for _ in range(cfg.max_passes):
        gain_db += cfg.target_lufs - current
        peak = float(np.abs(audio).max()) * 10.0 ** (gain_db / 20.0)
        peak_dbfs = 20.0 * np.log10(peak)
        if peak_dbfs > peak_limit:
            raise PeakLimitError(
                f"đưa về {cfg.target_lufs:g} LUFS cần gain {gain_db:+.1f} dB, đẩy đỉnh lên {peak_dbfs:.1f} dBFS "
                f"> {peak_limit:g} dBFS; không chuẩn hoá để tránh clip"
            )
        out = audio * 10.0 ** (gain_db / 20.0)
        current = measure_loudness(out, sample_rate)
        if abs(current - cfg.target_lufs) <= cfg.tolerance_lu:
            return out, LoudnessResult(input_lufs, current, gain_db, peak_dbfs)
    raise ValueError(
        f"sau {cfg.max_passes} lượt loudness vẫn là {current:.2f} LUFS, cách đích {cfg.target_lufs:g} LUFS "
        f"quá {cfg.tolerance_lu:g} LU"
    )


def normalize_file(src: Path, dst: Path, config: LoudnessConfig | None = None) -> LoudnessResult:
    """Đọc ``src``, chuẩn hoá, ghi ``dst`` (WAV, ``output_subtype``). Lỗi thì không ghi file."""
    cfg = config or load_loudness_config()
    audio, rate = read_float(src)
    out, result = normalize(audio, rate, cfg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out, rate, subtype=cfg.output_subtype, format="WAV")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vietvoice.data.loudness", description="Chuẩn hoá độ lớn.")
    parser.add_argument("src", type=Path, help="file WAV vào")
    parser.add_argument("dst", type=Path, help="file WAV ra")
    args = parser.parse_args(argv)
    result = normalize_file(args.src, args.dst)
    print(
        f"{args.src.name}: {result.input_lufs:.2f} LUFS -> {result.output_lufs:.2f} LUFS "
        f"(gain {result.gain_db:+.2f} dB, đỉnh {result.peak_dbfs:.2f} dBFS)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
