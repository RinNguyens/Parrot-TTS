"""Bước 1 pipeline: kiểm tra định dạng file phòng thu (đặc tả mục 6).

Đúng định dạng là: WAV (kể cả WAVE_FORMAT_EXTENSIBLE, soundfile báo "WAVEX"), đúng tần số lấy
mẫu, đúng độ sâu bit, mono, có mẫu âm thanh, và không clip.
Mọi thông số nằm trong ``qa.yaml`` (khối ``ingest``).

Clip: mẫu cách full scale không quá ``lsb_tolerance`` LSB (dương: 2^(n-1) - 1, âm: -2^(n-1)) là
mẫu chạm trần; ``min_run`` mẫu chạm trần liên tiếp trở lên mới tính là một đoạn clip. Một mẫu
lẻ chạm trần không tính clip ở đây, nhưng vẫn bị QA phiên thu loại vì đỉnh vượt -3 dBFS.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from parrot_tts.data.config import IngestConfig, load_qa_config
from parrot_tts.data.frames import run_lengths

# Độ sâu bit của các kiểu PCM số nguyên; kiểu khác (FLOAT, DOUBLE...) đo clip theo |x| >= 1.0.
_PCM_BITS = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}


@dataclass(frozen=True)
class FormatCheck:
    """Kết quả kiểm tra một file; ``reasons`` rỗng khi và chỉ khi ``ok``."""

    path: Path
    reasons: tuple[str, ...]
    container: str = ""
    sample_rate: int = 0
    channels: int = 0
    subtype: str = ""
    frames: int = 0
    clipped_runs: int = 0
    max_clip_run: int = 0

    @property
    def ok(self) -> bool:
        return not self.reasons


def load_ingest_config(path: Path | None = None) -> IngestConfig:
    return load_qa_config(path).ingest


def check_format(path: Path, config: IngestConfig | None = None) -> FormatCheck:
    """Kiểm tra định dạng và clip; file không đọc được thì trả kết quả lỗi kèm lý do."""
    cfg = config or load_ingest_config()
    if not path.is_file():
        raise FileNotFoundError(f"không thấy file: {path}")
    try:
        info = sf.info(str(path))
    except sf.LibsndfileError as err:
        return FormatCheck(path=path, reasons=(f"không đọc được file âm thanh: {err}",))

    want = cfg.format
    reasons: list[str] = []
    if info.format not in want.containers:
        reasons.append(f"định dạng {info.format}, cần {' hoặc '.join(want.containers)}")
    if info.samplerate != want.sample_rate:
        reasons.append(f"tần số lấy mẫu {info.samplerate} Hz, cần {want.sample_rate} Hz")
    if info.subtype != want.subtype:
        reasons.append(f"độ sâu bit {info.subtype}, cần {want.subtype}")
    if info.channels != want.channels:
        reasons.append(f"{info.channels} kênh, cần mono ({want.channels} kênh)")
    if info.frames == 0:
        reasons.append("file không có mẫu âm thanh")

    runs = _clip_runs_in_file(path, info.subtype, cfg) if info.frames else []
    if runs:
        reasons.append(
            f"có {len(runs)} đoạn clip (>= {cfg.clipping.min_run} mẫu liên tiếp sát full scale), "
            f"dài nhất {max(runs)} mẫu"
        )
    return FormatCheck(
        path=path,
        reasons=tuple(reasons),
        container=info.format,
        sample_rate=info.samplerate,
        channels=info.channels,
        subtype=info.subtype,
        frames=info.frames,
        clipped_runs=len(runs),
        max_clip_run=max(runs, default=0),
    )


def clipped_runs(samples: NDArray[np.int64], bits: int, lsb_tolerance: int) -> list[int]:
    """Độ dài các đoạn mẫu nguyên ``bits`` bit liên tiếp chạm trần (kể cả đoạn dài 1)."""
    top = 2 ** (bits - 1) - 1
    bottom = -(2 ** (bits - 1))
    return run_lengths((samples >= top - lsb_tolerance) | (samples <= bottom + lsb_tolerance))


def _clip_runs_in_file(path: Path, subtype: str, cfg: IngestConfig) -> list[int]:
    bits = _PCM_BITS.get(subtype)
    if bits is None:
        data, _ = sf.read(str(path), dtype="float64", always_2d=True)
        per_channel = [run_lengths(np.abs(data[:, c]) >= 1.0) for c in range(data.shape[1])]
    else:
        raw, _ = sf.read(str(path), dtype="int32", always_2d=True)
        # soundfile trả số nguyên căn trái trong int32; dịch phải về đúng ``bits`` bit.
        ints = raw.astype(np.int64) >> (32 - bits)
        tol = cfg.clipping.lsb_tolerance
        per_channel = [clipped_runs(ints[:, c], bits, tol) for c in range(ints.shape[1])]
    return [run for runs in per_channel for run in runs if run >= cfg.clipping.min_run]
