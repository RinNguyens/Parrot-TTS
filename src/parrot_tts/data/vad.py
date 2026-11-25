"""Bước 3 pipeline: cắt file phòng thu thành câu ở các khoảng lặng (đặc tả mục 6).

Hai lớp tách nhau:

- ``VoiceActivityDetector`` (Protocol): tìm các vùng có tiếng nói, trả ``SpeechRegion`` theo mẫu.
- ``segment_utterances``: gộp các vùng cách nhau dưới ``min_silence_s``, cắt ở khoảng lặng
  >= ``min_silence_s``, chừa ``pad_s`` lặng mỗi đầu (không vượt mép file, không lấn quá nửa
  khoảng lặng sang câu bên cạnh).

Đặc tả chọn Silero VAD, nhưng Silero cần torch, chưa có trong dự án. Tạm dùng ``EnergyVAD``
(mức RMS từng khung so với ngưỡng thích nghi theo từng file). Khi thêm torch ở P7, Silero VAD
chỉ cần một lớp có ``detect(audio, sample_rate) -> list[SpeechRegion]`` là cắm vào thay
``EnergyVAD``, phần gộp và đệm giữ nguyên.

Dòng lệnh: ``python -m parrot_tts.data.vad <vào.wav> <thư_mục_ra>``.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol, Self, runtime_checkable

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from pydantic import Field, model_validator

from parrot_tts.data.config import FrozenModel, load_yaml_model
from parrot_tts.data.frames import frame_power, power_db, read_float


class EnergyVADConfig(FrozenModel):
    frame_ms: float = Field(gt=0)
    noise_percentile: float = Field(gt=0, lt=100)
    loud_percentile: float = Field(gt=0, le=100)
    rel_threshold: float = Field(gt=0, lt=1)
    min_margin_db: float = Field(gt=0)
    db_floor: float


class VADConfig(FrozenModel):
    min_silence_s: float = Field(gt=0)
    pad_s: float = Field(ge=0)
    min_pad_s: float = Field(ge=0)
    output_subtype: str
    energy: EnergyVADConfig

    @model_validator(mode="after")
    def _pads_fit(self) -> Self:
        if not self.min_pad_s <= self.pad_s <= self.min_silence_s / 2:
            raise ValueError("cần min_pad_s <= pad_s <= min_silence_s / 2")
        return self


class SpeechRegion(NamedTuple):
    """Vùng tiếng nói ``[start, end)`` tính theo mẫu."""

    start: int
    end: int


@dataclass(frozen=True)
class UtteranceCut:
    """Một câu: đoạn cắt ``[start, end)`` gồm tiếng nói ``[speech_start, speech_end)`` và lặng đệm."""

    start: int
    end: int
    speech_start: int
    speech_end: int
    pad_ok: bool  # cả hai đầu có ít nhất min_pad_s lặng


@runtime_checkable
class VoiceActivityDetector(Protocol):
    def detect(self, audio: NDArray[np.float64], sample_rate: int) -> list[SpeechRegion]:
        """Các vùng tiếng nói theo thứ tự thời gian, không chồng nhau."""
        ...


class EnergyVAD:
    """VAD năng lượng: khung có mức >= ngưỡng là tiếng nói.

    Ngưỡng (dB) = nền + max(``min_margin_db``, ``rel_threshold`` × (to - nền)), với nền và to là
    phân vị ``noise_percentile`` và ``loud_percentile`` của mức khung trong chính file đó.
    Cần ít nhất 2% thời lượng là lặng để ước lượng nền đúng. Biên vùng được chỉnh về mức mẫu.
    """

    def __init__(self, config: EnergyVADConfig) -> None:
        self.config = config

    def detect(self, audio: NDArray[np.float64], sample_rate: int) -> list[SpeechRegion]:
        cfg = self.config
        mono = _mono(audio)
        if mono.size == 0:
            return []
        frame = max(1, round(cfg.frame_ms * sample_rate / 1000))
        level = power_db(frame_power(mono, frame), cfg.db_floor)
        noise = float(np.percentile(level, cfg.noise_percentile))
        loud = float(np.percentile(level, cfg.loud_percentile))
        threshold = noise + max(cfg.min_margin_db, cfg.rel_threshold * (loud - noise))
        edges = np.diff(np.concatenate(([0], (level >= threshold).astype(np.int8), [0])))
        starts = np.flatnonzero(edges == 1) * frame
        ends = np.flatnonzero(edges == -1) * frame
        return _refine_edges(mono, starts.tolist(), ends.tolist(), frame, threshold)


def _refine_edges(
    mono: NDArray[np.float64], starts: list[int], ends: list[int], frame: int, threshold_db: float
) -> list[SpeechRegion]:
    """Chỉnh biên vùng từ lưới khung về mức mẫu.

    Khung chỉ chứa một phần tiếng nói có thể bị xếp vào tiếng nói hoặc lặng, làm khoảng lặng đo
    theo khung lệch tới hai khung. Biên đầu là mẫu đầu tiên có |x| >= ngưỡng (đổi ra biên độ)
    trong khoảng một khung quanh biên khung; biên cuối là mẫu cuối cùng như vậy.
    """
    hot = np.abs(mono) >= 10.0 ** (threshold_db / 20.0)
    regions: list[SpeechRegion] = []
    for k, (a, b) in enumerate(zip(starts, ends, strict=True)):
        lo = max(a - frame, regions[-1].end if regions else 0)
        hits = np.flatnonzero(hot[lo : a + frame])
        start = lo + int(hits[0]) if hits.size else a
        hi_end = min(b + frame, starts[k + 1] if k + 1 < len(starts) else mono.size)
        hi_start = max(b - frame, start)
        hits = np.flatnonzero(hot[hi_start:hi_end])
        end = hi_start + int(hits[-1]) + 1 if hits.size else b
        regions.append(SpeechRegion(start, max(end, start + 1)))
    return regions


def load_vad_config(path: Path | None = None) -> VADConfig:
    return load_yaml_model(VADConfig, "vad", path)


def segment_utterances(
    audio: NDArray[np.float64], sample_rate: int, detector: VoiceActivityDetector, config: VADConfig | None = None
) -> list[UtteranceCut]:
    """Vùng tiếng nói -> các câu có lặng đệm hai đầu."""
    cfg = config or load_vad_config()
    n = _mono(audio).size
    min_gap = round(cfg.min_silence_s * sample_rate)
    pad = round(cfg.pad_s * sample_rate)
    min_pad = round(cfg.min_pad_s * sample_rate)

    merged: list[SpeechRegion] = []
    for region in detector.detect(audio, sample_rate):
        if not 0 <= region.start < region.end <= n or (merged and region.start < merged[-1].end):
            raise ValueError(f"VAD trả vùng không hợp lệ hoặc chồng nhau: {region}")
        if merged and region.start - merged[-1].end < min_gap:
            merged[-1] = SpeechRegion(merged[-1].start, region.end)
        else:
            merged.append(region)

    cuts = []
    for i, region in enumerate(merged):
        before = region.start - (merged[i - 1].end if i else 0)
        after = (merged[i + 1].start if i + 1 < len(merged) else n) - region.end
        # Khoảng lặng giữa hai câu chia đôi; nửa sau (lẻ một mẫu) thuộc câu sau.
        lead = min(pad, before - before // 2 if i else before)
        trail = min(pad, after // 2 if i + 1 < len(merged) else after)
        cuts.append(
            UtteranceCut(
                start=region.start - lead,
                end=region.end + trail,
                speech_start=region.start,
                speech_end=region.end,
                pad_ok=lead >= min_pad and trail >= min_pad,
            )
        )
    return cuts


def cut_file(
    src: Path, out_dir: Path, detector: VoiceActivityDetector, config: VADConfig | None = None
) -> list[tuple[Path, UtteranceCut]]:
    """Cắt ``src`` thành ``<tên>_0001.wav``... trong ``out_dir``; chép nguyên mẫu, không đổi mức."""
    cfg = config or load_vad_config()
    raw, rate = sf.read(str(src), dtype="int32")
    audio, _ = read_float(src)
    cuts = segment_utterances(audio, rate, detector, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k, cut in enumerate(cuts, start=1):
        path = out_dir / f"{src.stem}_{k:04d}.wav"
        sf.write(str(path), raw[cut.start : cut.end], rate, subtype=cfg.output_subtype, format="WAV")
        written.append((path, cut))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m parrot_tts.data.vad", description="Cắt file thu thành câu.")
    parser.add_argument("src", type=Path, help="file WAV vào")
    parser.add_argument("out_dir", type=Path, help="thư mục ghi các câu đã cắt")
    args = parser.parse_args(argv)
    cfg = load_vad_config()
    written = cut_file(args.src, args.out_dir, EnergyVAD(cfg.energy), cfg)
    short = sum(not cut.pad_ok for _, cut in written)
    print(f"{args.src.name}: {len(written)} câu, {short} câu thiếu lặng đệm (< {cfg.min_pad_s:g} s)")
    return 0


def _mono(audio: NDArray[np.float64]) -> NDArray[np.float64]:
    if audio.ndim == 2 and audio.shape[1] == 1:
        return audio[:, 0]
    if audio.ndim != 1:
        raise ValueError(f"VAD cần tín hiệu mono, gặp mảng hình {audio.shape}")
    return audio


if __name__ == "__main__":
    sys.exit(main())
