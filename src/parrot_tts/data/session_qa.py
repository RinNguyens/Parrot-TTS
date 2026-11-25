"""P4: QA chạy ngay sau mỗi phiên thu (đặc tả mục 5).

Mỗi file đạt khi: đúng định dạng và không clip (bước 1, ``ingest``), SNR >= 40 dB,
đỉnh <= -3 dBFS, nền ồn <= -60 dBFS. Ngưỡng nằm trong ``qa.yaml``.

Cách đo (mọi mức tính theo dBFS với full scale = 1,0, tức 20·log10(RMS); sin full scale là -3,01 dBFS):

- Khung: chia tín hiệu thành khung ``frame_ms`` không chồng lấn (bỏ khung cuối thiếu mẫu),
  mức mỗi khung = RMS của khung. File nhiều kênh được trộn trung bình về một kênh để đo.
- Nền ồn: bỏ các khung ở mức ``db_floor`` (số 0 tuyệt đối), rồi lấy các khung có mức <= phân vị
  ``noise_percentile`` (mặc định 2% khung nhỏ nhất), nền ồn = RMS gộp (trung bình công suất)
  của các khung đó. Cách này không cần biết trước chỗ lặng; đúng khi ít nhất 2% thời lượng là
  lặng (đặc tả: lặng 0,2–0,5 s đầu/cuối, tức >= 2,7% với câu 15 s).
  Với ồn dừng, kết quả thấp hơn RMS ồn thật khoảng 0,2–0,4 dB.
- Tiếng nói: các khung cao hơn nền ồn ít nhất ``speech_margin_db``; mức tiếng nói = RMS gộp
  các khung đó. Không có khung nào như vậy thì không tính được SNR và file bị loại.
- SNR = mức tiếng nói - nền ồn (dB).
- Đỉnh = max |mẫu| (sample peak, không phải true peak) trên mọi kênh.
- Số 0 tuyệt đối: phòng thu thật luôn có ồn, nên đoạn mẫu bằng đúng 0 dài hơn ``max_zero_run_ms``
  là dấu hiệu gate, khử ồn hoặc mất tín hiệu và bị loại.
- File ngắn hơn một khung thì không đo được và bị loại.

Dòng lệnh: ``python -m parrot_tts.data.session_qa <thư_mục_phiên> [--json out.json]``;
in bảng kết quả, trả mã thoát 1 nếu có file bị loại, 2 nếu sai tham số hoặc phiên rỗng.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from parrot_tts.data.config import LevelAnalysis, QAConfig, load_qa_config
from parrot_tts.data.frames import frame_power, power_db, read_float, run_lengths
from parrot_tts.data.ingest import check_format

__all__ = ["FileQA", "Levels", "SessionQA", "check_file", "load_qa_config", "main", "measure_levels", "run_session"]


@dataclass(frozen=True)
class Levels:
    peak_dbfs: float
    noise_floor_dbfs: float
    speech_dbfs: float | None
    snr_db: float | None
    longest_zero_run_s: float


@dataclass(frozen=True)
class FileQA:
    name: str
    passed: bool
    reasons: tuple[str, ...]
    format_ok: bool
    sample_rate: int
    subtype: str
    channels: int
    clipped_runs: int
    peak_dbfs: float | None
    noise_floor_dbfs: float | None
    snr_db: float | None


@dataclass(frozen=True)
class SessionQA:
    session: str
    files: tuple[FileQA, ...]

    @property
    def total(self) -> int:
        return len(self.files)

    @property
    def passed(self) -> int:
        return sum(f.passed for f in self.files)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def reject_rate(self) -> float:
        return self.failed / self.total

    def to_dict(self) -> dict[str, object]:
        return {
            "session": self.session,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "reject_rate": round(self.reject_rate, 4),
            "files": [asdict(f) for f in self.files],
        }


def measure_levels(audio: NDArray[np.float64], sample_rate: int, analysis: LevelAnalysis | None = None) -> Levels:
    """Đỉnh, nền ồn, mức tiếng nói và SNR theo định nghĩa ở đầu module."""
    cfg = analysis or load_qa_config().analysis
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    frame = max(1, round(cfg.frame_ms * sample_rate / 1000))
    if mono.size < frame:
        raise ValueError(f"tín hiệu ngắn hơn một khung đo ({mono.size} < {frame} mẫu)")
    power = frame_power(mono, frame)
    level = power_db(power, cfg.db_floor)
    peak_db = float(power_db(np.abs(audio).max() ** 2, cfg.db_floor))
    zero_run_s = max(run_lengths(mono == 0.0), default=0) / sample_rate

    live = level > cfg.db_floor
    if not live.any():
        return Levels(peak_db, cfg.db_floor, None, None, zero_run_s)
    live_power, live_level = power[live], level[live]
    quiet = live_power[live_level <= np.percentile(live_level, cfg.noise_percentile)]
    noise_db = float(power_db(quiet.mean(), cfg.db_floor))

    loud = live_power[live_level >= noise_db + cfg.speech_margin_db]
    if loud.size == 0:
        return Levels(peak_db, noise_db, None, None, zero_run_s)
    speech_db = float(power_db(loud.mean(), cfg.db_floor))
    return Levels(peak_db, noise_db, speech_db, speech_db - noise_db, zero_run_s)


def check_file(path: Path, config: QAConfig | None = None) -> FileQA:
    """QA một file: định dạng + clip (bước 1), rồi SNR, đỉnh, nền ồn."""
    cfg = config or load_qa_config()
    fmt = check_format(path, cfg.ingest)
    reasons = list(fmt.reasons)
    levels: Levels | None = None
    frame = max(1, round(cfg.analysis.frame_ms * fmt.sample_rate / 1000))
    if 0 < fmt.frames < frame:
        reasons.append(f"file quá ngắn để đo mức: {fmt.frames} mẫu < một khung {cfg.analysis.frame_ms:g} ms")
    elif fmt.frames:
        audio, rate = read_float(path)
        levels = measure_levels(audio, rate, cfg.analysis)
        reasons += _level_reasons(levels, cfg)
    return FileQA(
        name=path.name,
        passed=not reasons,
        reasons=tuple(reasons),
        format_ok=fmt.ok,
        sample_rate=fmt.sample_rate,
        subtype=fmt.subtype,
        channels=fmt.channels,
        clipped_runs=fmt.clipped_runs,
        peak_dbfs=None if levels is None else round(levels.peak_dbfs, 2),
        noise_floor_dbfs=None if levels is None else round(levels.noise_floor_dbfs, 2),
        snr_db=None if levels is None or levels.snr_db is None else round(levels.snr_db, 2),
    )


def run_session(session_dir: Path, config: QAConfig | None = None) -> SessionQA:
    """QA mọi file .wav (không đệ quy) trong thư mục phiên, theo thứ tự tên file."""
    if not session_dir.is_dir():
        raise FileNotFoundError(f"không thấy thư mục phiên: {session_dir}")
    wavs = sorted(p for p in session_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not wavs:
        raise ValueError(f"không có file .wav trong {session_dir}")
    cfg = config or load_qa_config()
    return SessionQA(session=session_dir.name, files=tuple(check_file(p, cfg) for p in wavs))


def format_table(summary: SessionQA) -> str:
    """Bảng kết quả tiếng Việt, một dòng mỗi file, cuối bảng là tỉ lệ loại."""
    header = ("File", "Định dạng", "SNR (dB)", "Đỉnh (dBFS)", "Nền ồn (dBFS)", "Clip", "Kết quả", "Lý do")
    rows: list[tuple[str, ...]] = [header]
    for f in summary.files:
        rows.append(
            (
                f.name,
                "đúng" if f.format_ok else "sai",
                _num(f.snr_db),
                _num(f.peak_dbfs),
                _num(f.noise_floor_dbfs),
                str(f.clipped_runs),
                "đạt" if f.passed else "loại",
                "; ".join(f.reasons),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header) - 1)]
    lines = ["  ".join([*(c.ljust(w) for c, w in zip(r[:-1], widths, strict=True)), r[-1]]).rstrip() for r in rows]
    lines.append(
        f"Phiên {summary.session}: {summary.total} file, đạt {summary.passed}, loại {summary.failed}, "
        f"tỉ lệ loại {summary.reject_rate:.1%}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m parrot_tts.data.session_qa", description="QA một phiên thu âm.")
    parser.add_argument("session_dir", type=Path, help="thư mục chứa các file .wav của phiên")
    parser.add_argument("--json", type=Path, dest="json_out", help="ghi kết quả ra file JSON")
    args = parser.parse_args(argv)
    try:
        summary = run_session(args.session_dir)
    except (FileNotFoundError, ValueError) as err:
        parser.error(str(err))  # in lỗi, thoát mã 2: khác mã 1 (có file bị loại)
    print(format_table(summary))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        args.json_out.write_text(text, encoding="utf-8")
    return 1 if summary.failed else 0


def _level_reasons(levels: Levels, cfg: QAConfig) -> list[str]:
    t = cfg.qa
    reasons = []
    if levels.snr_db is None:
        reasons.append(
            f"không có khung tiếng nói (cao hơn nền ồn >= {cfg.analysis.speech_margin_db:g} dB), không tính được SNR"
        )
    elif levels.snr_db < t.min_snr_db:
        reasons.append(f"SNR {levels.snr_db:.1f} dB < {t.min_snr_db:g} dB")
    if levels.peak_dbfs > t.max_peak_dbfs:
        reasons.append(f"đỉnh {levels.peak_dbfs:.1f} dBFS > {t.max_peak_dbfs:g} dBFS")
    if levels.noise_floor_dbfs > t.max_noise_floor_dbfs:
        reasons.append(f"nền ồn {levels.noise_floor_dbfs:.1f} dBFS > {t.max_noise_floor_dbfs:g} dBFS")
    if levels.longest_zero_run_s * 1000 > cfg.analysis.max_zero_run_ms:
        reasons.append(
            f"có đoạn lặng tuyệt đối (mẫu = 0) dài {levels.longest_zero_run_s * 1000:.0f} ms "
            f"> {cfg.analysis.max_zero_run_ms:g} ms, nghi gate / khử ồn / mất tín hiệu"
        )
    return reasons


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


if __name__ == "__main__":
    sys.exit(main())
