"""Biểu đồ cho luận văn: đường hội tụ loss và cột đối chứng thực nghiệm (Matplotlib + Seaborn).

Cần nhóm phụ thuộc ``viz`` (matplotlib, seaborn, pandas); module này không nằm trong bản sản phẩm.
Mọi hàm vẽ trả về ``matplotlib.figure.Figure`` tạo trực tiếp, không qua ``pyplot``, nên không
cần màn hình và không rò hình khi vẽ nhiều lần. Lưu bằng ``save_figure`` (PNG 300 dpi, PDF, SVG).

Đầu vào:

- Loss: bảng rộng, một cột trục hoành (mặc định ``step``) và mỗi đường loss một cột, như file CSV
  xuất từ Weights & Biases. Ô trống (val log thưa hơn train) bị bỏ theo từng cột.
- Đối chứng: bảng dài ``experiment, metric, value``, mỗi dòng một điểm. Nhiều dòng cùng
  (experiment, metric), ví dụ điểm từng câu, thì cột là trung bình và thanh sai số là khoảng tin cậy
  95% bootstrap (1000 lần, seed cố định nên vẽ lại ra đúng hình cũ). Tên chỉ số theo ``metrics.py``;
  ngưỡng đạt lấy từ ``metrics.yaml``.

Dòng lệnh::

    python -m parrot_tts.eval.plots loss log.csv -o loss.png -o loss.pdf [--smooth 0.6] [--log-y]
    python -m parrot_tts.eval.plots compare scores.csv -o compare.pdf [--metrics cer ter] [--order baseline E1]

Đầu vào sai thì hàm ném ``ValueError``; dòng lệnh in lỗi và trả mã thoát 2.
"""

import argparse
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from parrot_tts.eval.metrics import METRICS, CheckpointScores, Threshold, load_thresholds

PALETTE = "colorblind"
DPI = 300
N_BOOT = 1000
SEED = 0
THRESHOLD_LABEL = "ngưỡng"
SCORE_COLUMNS = ("experiment", "metric", "value")


@dataclass(frozen=True)
class MetricDisplay:
    """Cách hiển thị một chỉ số: tên, đơn vị, hệ số đổi từ giá trị lưu sang giá trị vẽ, chiều tốt."""

    label: str
    unit: str
    scale: float
    better: str  # "↓" thấp hơn tốt hơn, "↑" cao hơn tốt hơn, "→0" càng gần 0 càng tốt


DISPLAY: dict[str, MetricDisplay] = {
    "cer": MetricDisplay("CER", "%", 100.0, "↓"),
    "ter": MetricDisplay("TER", "%", 100.0, "↓"),
    "f0_rmse_st": MetricDisplay("F0 RMSE", "semitone", 1.0, "↓"),
    "creak_gap_pts": MetricDisplay("Chênh creak", "điểm %", 1.0, "→0"),
    "speaker_similarity": MetricDisplay("Speaker similarity", "cosine", 1.0, "↑"),
    "speech_rate_dev": MetricDisplay("Lệch tốc độ nói", "%", 100.0, "→0"),
}
assert set(DISPLAY) == set(METRICS), "DISPLAY phải có đúng các chỉ số trong metrics.py"


# ---------------------------------------------------------------- làm mượt


def _check_weight(weight: float) -> float:
    if not (math.isfinite(weight) and 0.0 <= weight < 1.0):
        raise ValueError(f"smooth phải nằm trong [0, 1), nhận {weight!r}")
    return float(weight)


def ema(values: npt.ArrayLike, weight: float) -> npt.NDArray[np.float64]:
    """Trung bình trượt mũ có khử lệch (như TensorBoard): s_t = w·s_{t-1} + (1-w)·x_t, chia (1-w^t).

    ``weight = 0`` trả lại nguyên chuỗi. Khử lệch để điểm đầu không bị kéo về 0.
    """
    w = _check_weight(weight)
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)
    s = 0.0
    for i, v in enumerate(x):
        s = w * s + (1.0 - w) * v
        out[i] = s / (1.0 - w ** (i + 1))
    return out


# ---------------------------------------------------------------- đường hội tụ loss


def _fmt(value: float) -> str:
    return f"{value:g}"


def plot_loss_curves(
    history: pd.DataFrame,
    *,
    x: str = "step",
    columns: Sequence[str] | None = None,
    smooth: float = 0.6,
    log_y: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (6.4, 4.0),
) -> Figure:
    """Đường hội tụ: mỗi cột loss một màu, đường mờ là giá trị thô, đường đậm là EMA ``smooth``.

    Cột đầu tiên có chữ "val" được đánh dấu điểm thấp nhất (giá trị thô), tức checkpoint tốt nhất.
    ``columns`` mặc định là mọi cột trừ ``x``.
    """
    weight = _check_weight(smooth)
    if x not in history.columns:
        raise ValueError(f"không có cột trục hoành {x!r}; các cột: {', '.join(map(str, history.columns))}")
    names = [str(c) for c in history.columns if c != x] if columns is None else list(columns)
    if not names:
        raise ValueError("không có cột loss nào để vẽ")

    series: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    for name in names:
        if name not in history.columns:
            raise ValueError(f"không có cột {name!r}")
        if not pd.api.types.is_numeric_dtype(history[name]):
            raise ValueError(f"cột {name!r} không phải số")
        part = history[[x, name]].dropna()
        if part.empty:
            raise ValueError(f"cột {name!r} không có giá trị nào")
        xs, ys = np.asarray(part[x], dtype=np.float64), np.asarray(part[name], dtype=np.float64)
        if log_y and (ys <= 0).any():
            raise ValueError(f"trục log cần loss dương, cột {name!r} có giá trị ≤ 0")
        series[name] = (xs, ys)

    best = next((n for n in names if "val" in n.lower()), None)
    with sns.axes_style("whitegrid"), sns.plotting_context("paper"):
        fig = Figure(figsize=figsize, layout="constrained")
        ax = fig.add_subplot()
        for (name, (xs, ys)), color in zip(series.items(), sns.color_palette(PALETTE, len(series)), strict=True):
            if weight > 0:
                ax.plot(xs, ys, color=color, alpha=0.25, linewidth=1.0, label=f"_{name} (thô)")
                ax.plot(xs, ema(ys, weight), color=color, linewidth=1.8, label=name)
            else:
                ax.plot(xs, ys, color=color, linewidth=1.6, label=name)
            if name == best:
                i = int(np.argmin(ys))
                ax.scatter([xs[i]], [ys[i]], color=color, zorder=3, s=24)
                ax.annotate(
                    f"min {ys[i]:.3g} @ {_fmt(xs[i])}",
                    (xs[i], ys[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize="small",
                )
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(x)
        ax.set_ylabel("loss (log)" if log_y else "loss")
        if title:
            ax.set_title(title)
        ax.legend(frameon=False)
    return fig


# ---------------------------------------------------------------- cột đối chứng


def scores_frame(checkpoints: Iterable[CheckpointScores]) -> pd.DataFrame:
    """Đổi các ``CheckpointScores`` sang bảng dài ``experiment, metric, value``; bỏ chỉ số chưa đo."""
    rows = [
        (cp.checkpoint, name, value) for cp in checkpoints for name, value in cp.scores().items() if value is not None
    ]
    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


def _validate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SCORE_COLUMNS if c not in scores.columns]
    if missing:
        raise ValueError(f"bảng điểm thiếu cột: {', '.join(missing)} (cần {', '.join(SCORE_COLUMNS)})")
    if scores.empty:
        raise ValueError("bảng điểm rỗng")
    data = pd.DataFrame({column: scores[column] for column in SCORE_COLUMNS})
    data["experiment"] = data["experiment"].astype(str)
    data["metric"] = data["metric"].astype(str)
    unknown = sorted(set(data["metric"]) - set(DISPLAY))
    if unknown:
        raise ValueError(f"chỉ số lạ: {', '.join(unknown)}; chỉ có {', '.join(METRICS)}")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if not np.isfinite(np.asarray(data["value"], dtype=np.float64)).all():
        raise ValueError("cột value có ô trống, không phải số hoặc không hữu hạn")
    return data


def _resolve_order(data: pd.DataFrame, order: Sequence[str] | None) -> list[str]:
    seen = list(dict.fromkeys(data["experiment"]))
    if order is None:
        return seen
    wanted = [str(e) for e in order]
    extra = [e for e in wanted if e not in seen]
    left_out = [e for e in seen if e not in wanted]
    if extra or left_out or len(set(wanted)) != len(wanted):
        raise ValueError(
            f"order phải liệt kê đúng mỗi thí nghiệm một lần; thừa: {', '.join(extra) or '-'}, "
            f"thiếu: {', '.join(left_out) or '-'}"
        )
    return wanted


def _draw_threshold(ax: Axes, threshold: Threshold, scale: float) -> None:
    limit = threshold.limit * scale
    levels = [-limit, limit] if threshold.op == "abs_max" else [limit]
    for level in levels:
        ax.axhline(level, color="0.25", linestyle="--", linewidth=1.0, label=THRESHOLD_LABEL)


def plot_experiment_comparison(
    scores: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    thresholds: Mapping[str, Threshold] | None = None,
    order: Sequence[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Cột đối chứng: mỗi chỉ số một ô, mỗi thí nghiệm một cột, nét đứt là ngưỡng đạt.

    ``metrics`` mặc định là các chỉ số có trong bảng, theo thứ tự của ``metrics.py``.
    ``order`` mặc định là thứ tự thí nghiệm xuất hiện lần đầu trong bảng.
    """
    data = _validate_scores(scores)
    present = set(data["metric"])
    if metrics is None:
        chosen = [m for m in METRICS if m in present]
    else:
        chosen = list(metrics)
        unknown = [m for m in chosen if m not in DISPLAY]
        if unknown:
            raise ValueError(f"chỉ số lạ: {', '.join(unknown)}; chỉ có {', '.join(METRICS)}")
        absent = [m for m in chosen if m not in present]
        if absent:
            raise ValueError(f"không có điểm cho chỉ số: {', '.join(absent)}")
    experiments = _resolve_order(data, order)
    table = load_thresholds() if thresholds is None else thresholds
    colors = dict(zip(experiments, sns.color_palette(PALETTE, len(experiments)), strict=True))

    ncols = min(3, len(chosen))
    nrows = math.ceil(len(chosen) / ncols)
    with sns.axes_style("whitegrid"), sns.plotting_context("paper"):
        fig = Figure(figsize=(3.2 * ncols, 3.0 * nrows), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
        for ax, metric in zip(axes, chosen, strict=False):
            display = DISPLAY[metric]
            part = data[data["metric"] == metric].assign(value=lambda d, s=display.scale: d["value"] * s)
            sns.barplot(
                data=part,
                x="experiment",
                y="value",
                hue="experiment",
                order=experiments,
                hue_order=experiments,
                palette=colors,
                legend=False,
                errorbar=("ci", 95),
                n_boot=N_BOOT,
                seed=SEED,
                capsize=0.2,
                err_kws={"linewidth": 1.0, "color": "0.2"},
                ax=ax,
            )
            if metric in table:
                _draw_threshold(ax, table[metric], display.scale)
            ax.set_title(f"{display.label} ({display.better})")
            ax.set_xlabel("")
            ax.set_ylabel(display.unit)
            if len(experiments) > 4:
                ax.tick_params(axis="x", labelrotation=30)
        for ax in axes[len(chosen) :]:
            ax.set_visible(False)
        handles = [line for line in axes[0].lines if line.get_label() == THRESHOLD_LABEL][:1]
        if handles:
            axes[0].legend(handles=handles, frameon=False, fontsize="small")
        if title:
            fig.suptitle(title)
    return fig


# ---------------------------------------------------------------- lưu và dòng lệnh


def save_figure(fig: Figure, paths: Iterable[Path], *, dpi: int = DPI) -> list[Path]:
    """Lưu hình theo đuôi file (png, pdf, svg...). PDF/SVG giữ chữ dạng font để sửa được trong luận văn."""
    written: list[Path] = []
    with matplotlib.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi)
            written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m parrot_tts.eval.plots", description="Vẽ biểu đồ cho luận văn.")
    sub = parser.add_subparsers(dest="command", required=True)

    loss = sub.add_parser("loss", help="đường hội tụ loss từ CSV dạng rộng")
    loss.add_argument("csv", type=Path, help="CSV: cột trục hoành và các cột loss")
    loss.add_argument("--x", default="step", help="cột trục hoành (mặc định: step)")
    loss.add_argument("--columns", nargs="+", help="các cột loss (mặc định: mọi cột trừ trục hoành)")
    loss.add_argument("--smooth", type=float, default=0.6, help="hệ số EMA trong [0, 1) (mặc định: 0.6)")
    loss.add_argument("--log-y", action="store_true", help="trục tung thang log")

    compare = sub.add_parser("compare", help="cột đối chứng từ CSV experiment,metric,value")
    compare.add_argument("csv", type=Path, help="CSV dạng dài: experiment, metric, value")
    compare.add_argument("--metrics", nargs="+", help=f"chỉ số cần vẽ, trong: {', '.join(METRICS)}")
    compare.add_argument("--order", nargs="+", help="thứ tự thí nghiệm trên trục hoành")

    for p in (loss, compare):
        p.add_argument("-o", "--output", type=Path, action="append", required=True, help="file ra, lặp được")
        p.add_argument("--title", help="tiêu đề hình")
        p.add_argument("--dpi", type=int, default=DPI, help=f"độ phân giải ảnh raster (mặc định: {DPI})")

    args = parser.parse_args(argv)
    try:
        frame = pd.read_csv(args.csv)
        if args.command == "loss":
            fig = plot_loss_curves(
                frame, x=args.x, columns=args.columns, smooth=args.smooth, log_y=args.log_y, title=args.title
            )
        else:
            fig = plot_experiment_comparison(frame, metrics=args.metrics, order=args.order, title=args.title)
    except (OSError, ValueError) as error:  # pandas.errors.ParserError là ValueError
        print(f"lỗi: {error}", file=sys.stderr)
        return 2
    for path in save_figure(fig, args.output, dpi=args.dpi):
        print(f"đã ghi {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
