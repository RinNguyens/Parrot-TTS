"""Chỉ số tự động chấm checkpoint (đặc tả mục 10, kế hoạch P6).

Các hàm ở đây chỉ tính trên đầu vào đã có sẵn: bản ASR, nhãn thanh từng âm tiết, mảng F0,
cờ creak, embedding người nói. Chạy ASR, MFA, bộ phân loại thanh, UTMOS nằm ở chỗ khác.

Quy ước:
- CER, WER, TER, độ lệch tốc độ nói trả về tỉ lệ (0.03 = 3%); chênh creak trả về điểm %.
- Văn bản được chuẩn hoá NFC, viết thường, gộp khoảng trắng trước khi so. Dấu câu giữ nguyên.
- CER theo quy ước của jiwer: dấu cách là một ký tự.
- Ngưỡng đạt nằm trong ``metrics.yaml`` cạnh file này.

Đầu vào sai (văn bản gốc rỗng, thanh ngoài 1..8, vector 0, ...) thì ném ``ValueError``.
"""

import math
import unicodedata
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import yaml

THRESHOLDS_PATH = Path(__file__).with_name("metrics.yaml")
N_TONES = 8

ArrayLike = npt.ArrayLike
Op = Literal["max", "min", "abs_max"]
OPS: tuple[str, ...] = get_args(Op)

# ---------------------------------------------------------------- khoảng cách sửa (Levenshtein)


def _distance_table[T: Hashable](ref: Sequence[T], hyp: Sequence[T]) -> list[list[int]]:
    """Bảng quy hoạch động: ô [i][j] = số lỗi ít nhất giữa ref[:i] và hyp[:j]."""
    table = [list(range(len(hyp) + 1))]
    for i in range(1, len(ref) + 1):
        prev, row = table[-1], [i]
        for j in range(1, len(hyp) + 1):
            sub = prev[j - 1] + (ref[i - 1] != hyp[j - 1])
            row.append(min(sub, prev[j] + 1, row[j - 1] + 1))
        table.append(row)
    return table


def _edit_distance[T: Hashable](ref: Sequence[T], hyp: Sequence[T]) -> int:
    return _distance_table(ref, hyp)[-1][-1]


def _align[T: Hashable](ref: Sequence[T], hyp: Sequence[T]) -> list[tuple[T | None, T | None]]:
    """Căn hai chuỗi theo khoảng cách sửa. ``(r, None)`` là bớt, ``(None, h)`` là thêm.

    Khi hoà điểm, ưu tiên khớp/thay, rồi bớt, rồi thêm.
    """
    table = _distance_table(ref, hyp)
    pairs: list[tuple[T | None, T | None]] = []
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            pairs.append((ref[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hyp[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------- CER, WER


def normalize_text(text: str) -> str:
    """NFC, viết thường, gộp mọi khoảng trắng thành một dấu cách, bỏ khoảng trắng hai đầu."""
    lowered = unicodedata.normalize("NFC", text).lower()
    return " ".join(unicodedata.normalize("NFC", lowered).split())


def cer(reference: str, hypothesis: str) -> float:
    """Tỉ lệ lỗi ký tự = (thay + thêm + bớt) / số ký tự bản gốc. Dấu cách tính là ký tự.

    Có thể lớn hơn 1 khi bản nhận dạng thêm nhiều ký tự.
    """
    ref = normalize_text(reference)
    if not ref:
        raise ValueError("văn bản gốc rỗng, không tính được CER")
    return _edit_distance(ref, normalize_text(hypothesis)) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Tỉ lệ lỗi từ (âm tiết, vì tiếng Việt viết tách) sau cùng bước chuẩn hoá như CER."""
    ref = normalize_text(reference).split()
    if not ref:
        raise ValueError("văn bản gốc rỗng, không tính được WER")
    return _edit_distance(ref, normalize_text(hypothesis).split()) / len(ref)


# ---------------------------------------------------------------- TER, ma trận nhầm lẫn


def _check_tone(value: object, where: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{where}: thanh phải là số nguyên 1..{N_TONES}, nhận {value!r}")
    tone = int(value)
    if not 1 <= tone <= N_TONES:
        raise ValueError(f"{where}: thanh {tone} ngoài 1..{N_TONES}")
    return tone


def _tones(values: Iterable[object], where: str) -> list[int]:
    return [_check_tone(v, where) for v in values]


def align_tones(ref_tones: Iterable[object], hyp_tones: Iterable[object]) -> list[tuple[int | None, int | None]]:
    """Căn nhãn thanh đích với nhãn thanh nhận được theo âm tiết (khoảng cách sửa)."""
    return _align(_tones(ref_tones, "nhãn đích"), _tones(hyp_tones, "nhãn nhận được"))


def tone_error_rate(ref_tones: Iterable[object], hyp_tones: Iterable[object]) -> float:
    """TER = (thay + thêm + bớt âm tiết) / số âm tiết đích. Hai chuỗi có thể khác độ dài."""
    ref = _tones(ref_tones, "nhãn đích")
    if not ref:
        raise ValueError("chuỗi thanh đích rỗng, không tính được TER")
    return _edit_distance(ref, _tones(hyp_tones, "nhãn nhận được")) / len(ref)


def tone_confusion_matrix(pairs: Iterable[tuple[object, object]]) -> npt.NDArray[np.int64]:
    """Ma trận 8 × 8: hàng là thanh đích, cột là thanh nhận được, chỉ số = thanh - 1.

    Nhận đầu ra của ``align_tones``; cặp có một vế ``None`` (âm tiết thêm hoặc bớt)
    không có ô trong ma trận nên được bỏ qua. Cặp ``(None, None)`` là lỗi.
    """
    matrix = np.zeros((N_TONES, N_TONES), dtype=np.int64)
    for ref, hyp in pairs:
        if ref is None and hyp is None:
            raise ValueError("cặp (None, None) không hợp lệ")
        if ref is None or hyp is None:
            _check_tone(hyp if ref is None else ref, "cặp thêm/bớt")
            continue
        matrix[_check_tone(ref, "nhãn đích") - 1, _check_tone(hyp, "nhãn nhận được") - 1] += 1
    return matrix


# ---------------------------------------------------------------- F0, DTW


def _as_1d(values: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} phải là mảng 1 chiều, nhận {array.ndim} chiều")
    return array


def hz_to_semitones(f0_hz: ArrayLike, baseline_hz: float) -> npt.NDArray[np.float64]:
    """Đổi Hz sang semitone so với mức nền của người nói: 12·log2(f / nền)."""
    if not (math.isfinite(baseline_hz) and baseline_hz > 0):
        raise ValueError(f"baseline F0 phải dương, nhận {baseline_hz!r}")
    f0 = np.asarray(f0_hz, dtype=np.float64)
    if not np.all(np.isfinite(f0) & (f0 > 0)):
        raise ValueError("F0 có giá trị không dương; lọc khung vô thanh trước khi đổi")
    return 12.0 * np.log2(f0 / baseline_hz)


def dtw(a: ArrayLike, b: ArrayLike) -> tuple[float, list[tuple[int, int]]]:
    """DTW thường, chi phí |a_i - b_j|, bước (1,0) (0,1) (1,1), không giới hạn dải.

    Trả về (tổng chi phí nhỏ nhất, đường căn từ (0, 0) đến (len(a)-1, len(b)-1)).
    """
    x, y = _as_1d(a, "a"), _as_1d(b, "b")
    if x.size == 0 or y.size == 0:
        raise ValueError("DTW cần hai chuỗi không rỗng")
    cost = np.abs(x[:, None] - y[None, :]).tolist()
    n, m = x.size, y.size
    inf = math.inf
    acc = [[inf] * (m + 1) for _ in range(n + 1)]
    acc[0][0] = 0.0
    for i in range(1, n + 1):
        prev, row, c = acc[i - 1], acc[i], cost[i - 1]
        for j in range(1, m + 1):
            row[j] = c[j - 1] + min(prev[j - 1], prev[j], row[j - 1])

    path = [(n - 1, m - 1)]
    i, j = n, m
    while (i, j) != (1, 1):
        # Hoà điểm thì ưu tiên đi chéo.
        steps = [(acc[i - 1][j - 1], i - 1, j - 1), (acc[i - 1][j], i - 1, j), (acc[i][j - 1], i, j - 1)]
        _, i, j = min(steps, key=lambda s: s[0])
        path.append((i - 1, j - 1))
    path.reverse()
    return float(acc[n][m]), path


def _voiced(f0_hz: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    f0 = _as_1d(f0_hz, name)
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size == 0:
        raise ValueError(f"{name} không có khung hữu thanh nào (mọi F0 ≤ 0 hoặc NaN)")
    return voiced


def f0_rmse_semitones(ref_f0_hz: ArrayLike, hyp_f0_hz: ArrayLike, ref_baseline_hz: float) -> float:
    """RMSE (semitone) giữa F0 bản thật và bản tổng hợp, sau khi căn bằng DTW.

    Cả hai đổi sang semitone theo cùng mức nền của người nói thật. Khung vô thanh
    (F0 ≤ 0 hoặc NaN) bị bỏ, các khung hữu thanh còn lại nối liền nhau rồi mới căn.
    Dùng cho một âm tiết (hoặc một đoạn); gộp nhiều âm tiết là việc của nơi gọi.
    """
    ref = hz_to_semitones(_voiced(ref_f0_hz, "F0 bản thật"), ref_baseline_hz)
    hyp = hz_to_semitones(_voiced(hyp_f0_hz, "F0 bản tổng hợp"), ref_baseline_hz)
    _, path = dtw(ref, hyp)
    rows, cols = zip(*path, strict=True)
    diff = ref[list(rows)] - hyp[list(cols)]
    return float(np.sqrt(np.mean(diff**2)))


# ---------------------------------------------------------------- creak, similarity, tốc độ


def _creak_percent(flags: Iterable[object], name: str) -> float:
    values: list[bool] = []
    for flag in flags:
        if not isinstance(flag, bool | np.bool_ | int | np.integer) or flag not in (0, 1):
            raise ValueError(f"{name}: cờ creak phải là True/False hoặc 0/1, nhận {flag!r}")
        values.append(bool(flag))
    if not values:
        raise ValueError(f"{name}: không có âm tiết ngã/nặng nào")
    return 100.0 * sum(values) / len(values)


def creak_rate_gap(ref_flags: Iterable[object], hyp_flags: Iterable[object]) -> float:
    """Chênh tỉ lệ creak (điểm %) = % ở bản tổng hợp − % ở bản thật, có dấu.

    Mỗi cờ ứng với một âm tiết ngã/nặng (thanh 3, 6, 8); nơi gọi tự lọc âm tiết.
    """
    return _creak_percent(hyp_flags, "bản tổng hợp") - _creak_percent(ref_flags, "bản thật")


def speaker_similarity(emb_a: ArrayLike, emb_b: ArrayLike) -> float:
    """Cosine giữa hai embedding người nói (1 chiều, cùng độ dài)."""
    a, b = _as_1d(emb_a, "embedding a"), _as_1d(emb_b, "embedding b")
    if a.shape != b.shape:
        raise ValueError(f"hai embedding khác độ dài: {a.size} và {b.size}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("embedding có NaN hoặc vô cực")
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("embedding là vector 0, không tính được cosine")
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def speech_rate(n_syllables: int, duration_s: float) -> float:
    """Tốc độ nói, âm tiết mỗi giây."""
    if n_syllables < 0:
        raise ValueError(f"số âm tiết âm: {n_syllables}")
    if not (math.isfinite(duration_s) and duration_s > 0):
        raise ValueError(f"độ dài phải dương, nhận {duration_s!r} giây")
    return n_syllables / duration_s


def speech_rate_deviation(synth_rate: float, real_rate: float) -> float:
    """Độ lệch tốc độ nói có dấu, dạng tỉ lệ: (tổng hợp − thật) / thật. Dương là nói nhanh hơn."""
    if not (math.isfinite(real_rate) and real_rate > 0):
        raise ValueError(f"tốc độ bản thật phải dương, nhận {real_rate!r}")
    if not (math.isfinite(synth_rate) and synth_rate >= 0):
        raise ValueError(f"tốc độ bản tổng hợp không hợp lệ: {synth_rate!r}")
    return (synth_rate - real_rate) / real_rate


# ---------------------------------------------------------------- ngưỡng và tổng hợp checkpoint


@dataclass(frozen=True)
class Threshold:
    """Một ngưỡng: ``max`` điểm ≤ limit, ``min`` điểm ≥ limit, ``abs_max`` |điểm| ≤ limit."""

    metric: str
    op: Op
    limit: float

    def check(self, value: float) -> bool:
        if self.op == "max":
            return value <= self.limit
        if self.op == "min":
            return value >= self.limit
        return abs(value) <= self.limit


@dataclass(frozen=True)
class CheckpointScores:
    """Điểm tự động của một checkpoint. ``None`` là chưa đo."""

    checkpoint: str
    cer: float | None = None
    ter: float | None = None
    f0_rmse_st: float | None = None
    creak_gap_pts: float | None = None
    speaker_similarity: float | None = None
    speech_rate_dev: float | None = None

    def scores(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in METRICS}

    def missing(self) -> list[str]:
        """Các chỉ số chưa đo."""
        return [name for name, value in self.scores().items() if value is None]

    def verdicts(self, thresholds: Mapping[str, Threshold] | None = None) -> dict[str, bool | None]:
        """Đạt/không đạt từng chỉ số; ``None`` nếu chưa đo. Mặc định đọc ``metrics.yaml``."""
        table = load_thresholds() if thresholds is None else thresholds
        result: dict[str, bool | None] = {}
        for name, value in self.scores().items():
            if name not in table:
                raise ValueError(f"không có ngưỡng cho chỉ số {name!r}")
            if value is None:
                result[name] = None
            elif not math.isfinite(value):
                raise ValueError(f"{self.checkpoint}: điểm {name} không hợp lệ ({value!r})")
            else:
                result[name] = table[name].check(value)
        return result

    def passed(self, thresholds: Mapping[str, Threshold] | None = None) -> bool:
        """Chỉ đạt khi mọi chỉ số đã đo và đều đạt; thiếu một chỉ số là chưa đạt."""
        return all(v is True for v in self.verdicts(thresholds).values())


METRICS: tuple[str, ...] = tuple(f.name for f in fields(CheckpointScores) if f.name != "checkpoint")


def load_thresholds(path: Path | None = None) -> dict[str, Threshold]:
    """Đọc ngưỡng từ YAML (mặc định ``metrics.yaml``). File phải có đủ và chỉ có các chỉ số đã biết."""
    source = THRESHOLDS_PATH if path is None else path
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source.name} phải là bảng chỉ số -> ngưỡng")
    unknown = set(raw) - set(METRICS)
    if unknown:
        raise ValueError(f"{source.name} có chỉ số lạ: {', '.join(sorted(map(str, unknown)))}")
    missing = set(METRICS) - set(raw)
    if missing:
        raise ValueError(f"{source.name} thiếu ngưỡng: {', '.join(sorted(missing))}")
    thresholds: dict[str, Threshold] = {}
    for name in METRICS:
        spec = raw[name]
        if not isinstance(spec, dict) or len(spec) != 1:
            raise ValueError(f"{source.name}: {name} phải có đúng một phép so sánh ({', '.join(OPS)})")
        ((op, limit),) = spec.items()
        if op not in OPS:
            raise ValueError(f"{source.name}: {name} có phép lạ {op!r}; chỉ có {', '.join(OPS)}")
        if isinstance(limit, bool) or not isinstance(limit, int | float) or not math.isfinite(limit):
            raise ValueError(f"{source.name}: ngưỡng {name} phải là số, nhận {limit!r}")
        thresholds[name] = Threshold(metric=name, op=op, limit=float(limit))
    return thresholds
