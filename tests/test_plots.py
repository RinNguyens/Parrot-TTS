"""Biểu đồ cho luận văn: đường hội tụ loss và cột đối chứng thực nghiệm.

API: ``vietvoice.eval.plots``. Test kiểm tra nội dung hình (số trục, đường, ngưỡng, chiều cao cột),
không so ảnh từng điểm ảnh.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from vietvoice.eval.metrics import METRICS, CheckpointScores, Threshold
from vietvoice.eval.plots import (
    ema,
    main,
    plot_experiment_comparison,
    plot_loss_curves,
    scores_frame,
)

# ---------------------------------------------------------------- EMA


def test_ema_weight_zero_is_identity():
    values = [3.0, 1.0, 2.0]
    assert ema(values, 0.0).tolist() == values


def test_ema_is_debiased_so_first_value_is_kept():
    # Không khử lệch thì giá trị đầu bị kéo về 0: 0.9·0 + 0.1·10 = 1.
    assert ema([10.0, 10.0, 10.0], 0.9).tolist() == pytest.approx([10.0, 10.0, 10.0])


def test_ema_known_values():
    # s1 = 0.5·0 + 0.5·2 = 1, khử lệch /0.5 -> 2; s2 = 0.5·1 + 0.5·4 = 2.5, /0.75 -> 3.333...
    assert ema([2.0, 4.0], 0.5).tolist() == pytest.approx([2.0, 10.0 / 3.0])


@pytest.mark.parametrize("weight", [-0.1, 1.0, 1.5, math.nan])
def test_ema_rejects_weight_outside_range(weight: float):
    with pytest.raises(ValueError, match="smooth"):
        ema([1.0, 2.0], weight)


# ---------------------------------------------------------------- loss


def _history() -> pd.DataFrame:
    steps = np.arange(0, 1000, 10)
    train = 2.0 * np.exp(-steps / 300) + 0.1
    val = np.full(steps.shape, np.nan)
    val[::10] = 2.2 * np.exp(-steps[::10] / 300) + 0.15  # val log thưa hơn train
    val[-1] = 0.4  # val tăng lại ở cuối: điểm thấp nhất không phải điểm cuối
    return pd.DataFrame({"step": steps, "train_loss": train, "val_loss": val})


def test_loss_returns_figure_with_raw_and_smoothed_line_per_column():
    fig = plot_loss_curves(_history(), smooth=0.6)
    assert isinstance(fig, Figure)
    (ax,) = fig.axes
    assert len(ax.lines) == 4  # 2 cột × (thô + làm mượt)


def test_loss_without_smoothing_draws_one_line_per_column():
    (ax,) = plot_loss_curves(_history(), smooth=0.0).axes
    assert len(ax.lines) == 2


def test_loss_val_nan_rows_are_dropped_not_plotted_as_gaps():
    (ax,) = plot_loss_curves(_history(), columns=["val_loss"], smooth=0.0).axes
    (line,) = ax.lines
    assert len(np.asarray(line.get_xdata())) == 11  # 10 điểm thưa + điểm cuối
    assert not np.isnan(np.asarray(line.get_ydata(), dtype=float)).any()


def test_loss_marks_minimum_of_validation_column():
    history = _history()
    best = history.loc[history["val_loss"].idxmin()]
    (ax,) = plot_loss_curves(history).axes
    texts = [t.get_text() for t in ax.texts]
    assert any(f"{int(best['step'])}" in t and "min" in t for t in texts), texts


def test_loss_columns_default_to_every_column_except_x():
    history = _history().assign(lr=0.001)
    (ax,) = plot_loss_curves(history, smooth=0.0).axes
    assert len(ax.lines) == 3


def test_loss_custom_x_column_and_labels():
    history = _history().rename(columns={"step": "epoch"})
    (ax,) = plot_loss_curves(history, x="epoch", title="Hội tụ").axes
    assert ax.get_xlabel() == "epoch"
    assert ax.get_title() == "Hội tụ"


def test_loss_log_scale():
    (ax,) = plot_loss_curves(_history(), log_y=True).axes
    assert ax.get_yscale() == "log"


def test_loss_log_scale_rejects_non_positive():
    history = _history()
    history.loc[0, "train_loss"] = 0.0
    with pytest.raises(ValueError, match="log"):
        plot_loss_curves(history, log_y=True)


def test_loss_missing_x_column():
    with pytest.raises(ValueError, match="step"):
        plot_loss_curves(_history().drop(columns="step"))


def test_loss_unknown_column():
    with pytest.raises(ValueError, match="mel_loss"):
        plot_loss_curves(_history(), columns=["mel_loss"])


def test_loss_column_all_nan():
    with pytest.raises(ValueError, match="val_loss"):
        plot_loss_curves(_history().assign(val_loss=np.nan))


def test_loss_non_numeric_column():
    with pytest.raises(ValueError, match="note"):
        plot_loss_curves(_history().assign(note="a"), columns=["note"])


def test_loss_no_loss_column():
    with pytest.raises(ValueError, match="không có cột"):
        plot_loss_curves(pd.DataFrame({"step": _history()["step"]}))


# ---------------------------------------------------------------- đối chứng


def _scores() -> pd.DataFrame:
    rows = []
    for exp, cer_values, sim in [("baseline", [0.06, 0.08], 0.70), ("E1", [0.02, 0.03], 0.80)]:
        rows += [{"experiment": exp, "metric": "cer", "value": v} for v in cer_values]
        rows.append({"experiment": exp, "metric": "speaker_similarity", "value": sim})
    return pd.DataFrame(rows)


def _threshold_levels(ax) -> list[float]:
    return sorted(float(np.asarray(line.get_ydata())[0]) for line in ax.lines if line.get_label() == "ngưỡng")


def _bar_heights(ax) -> list[float]:
    return [p.get_height() for p in ax.patches if p.get_height() == p.get_height()]


def test_compare_one_axes_per_metric_in_canonical_order():
    fig = plot_experiment_comparison(_scores())
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axes) == 2
    assert "CER" in axes[0].get_title()
    assert "similarity" in axes[1].get_title().lower()


def test_compare_bar_is_mean_scaled_to_display_unit():
    ax = plot_experiment_comparison(_scores(), metrics=["cer"]).axes[0]
    assert _bar_heights(ax) == pytest.approx([7.0, 2.5])  # tỉ lệ -> %


def test_compare_keeps_experiment_order_of_first_appearance():
    ax = plot_experiment_comparison(_scores(), metrics=["cer"]).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["baseline", "E1"]


def test_compare_explicit_order():
    ax = plot_experiment_comparison(_scores(), metrics=["cer"], order=["E1", "baseline"]).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["E1", "baseline"]
    assert _bar_heights(ax) == pytest.approx([2.5, 7.0])


def test_compare_draws_threshold_line_from_metrics_yaml():
    ax = plot_experiment_comparison(_scores(), metrics=["cer"]).axes[0]
    levels = _threshold_levels(ax)
    assert levels == pytest.approx([3.0])  # 0.03 trong metrics.yaml, hiển thị %


def test_compare_abs_max_threshold_draws_both_sides():
    scores = pd.DataFrame({"experiment": ["E1"], "metric": ["speech_rate_dev"], "value": [-0.05]})
    ax = plot_experiment_comparison(scores).axes[0]
    levels = _threshold_levels(ax)
    assert levels == pytest.approx([-10.0, 10.0])


def test_compare_custom_thresholds():
    thresholds = {name: Threshold(name, "max", 1.0) for name in METRICS}
    ax = plot_experiment_comparison(_scores(), metrics=["cer"], thresholds=thresholds).axes[0]
    levels = _threshold_levels(ax)
    assert levels == pytest.approx([100.0])


def _error_bars(ax) -> list[np.ndarray]:
    """Đường thanh sai số có ít nhất một điểm hữu hạn (seaborn vẽ đường toàn NaN khi chỉ có 1 mẫu)."""
    lines = [np.asarray(line.get_ydata(), dtype=float) for line in ax.lines if line.get_label() != "ngưỡng"]
    return [y for y in lines if np.isfinite(y).any()]


def test_compare_error_bars_only_with_repeated_samples():
    fig = plot_experiment_comparison(_scores())
    cer_ax, sim_ax = fig.axes[:2]
    assert _error_bars(cer_ax), "CER có 2 mẫu mỗi thí nghiệm, phải có thanh sai số"
    assert not _error_bars(sim_ax)


def test_compare_bootstrap_is_reproducible():
    first = _error_bars(plot_experiment_comparison(_scores()).axes[0])
    second = _error_bars(plot_experiment_comparison(_scores()).axes[0])
    assert len(first) == len(second) > 0
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_compare_unknown_metric():
    scores = _scores().assign(metric="mos")
    with pytest.raises(ValueError, match="mos"):
        plot_experiment_comparison(scores)


def test_compare_missing_columns():
    with pytest.raises(ValueError, match="experiment"):
        plot_experiment_comparison(_scores().drop(columns="experiment"))


def test_compare_nan_value():
    scores = _scores()
    scores.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="value"):
        plot_experiment_comparison(scores)


def test_compare_empty():
    with pytest.raises(ValueError, match="rỗng"):
        plot_experiment_comparison(_scores().iloc[0:0])


def test_compare_requested_metric_absent():
    with pytest.raises(ValueError, match="ter"):
        plot_experiment_comparison(_scores(), metrics=["ter"])


def test_compare_order_must_match_experiments():
    with pytest.raises(ValueError, match="E9"):
        plot_experiment_comparison(_scores(), order=["baseline", "E9"])


# ---------------------------------------------------------------- CheckpointScores -> bảng


def test_scores_frame_skips_unmeasured_metrics():
    frame = scores_frame([CheckpointScores("E1", cer=0.02, ter=0.01), CheckpointScores("E2", cer=0.04)])
    assert frame.to_dict("records") == [
        {"experiment": "E1", "metric": "cer", "value": 0.02},
        {"experiment": "E1", "metric": "ter", "value": 0.01},
        {"experiment": "E2", "metric": "cer", "value": 0.04},
    ]


def test_scores_frame_feeds_comparison_plot():
    frame = scores_frame([CheckpointScores("E1", cer=0.02), CheckpointScores("E2", cer=0.04)])
    ax = plot_experiment_comparison(frame).axes[0]
    assert _bar_heights(ax) == pytest.approx([2.0, 4.0])


# ---------------------------------------------------------------- dòng lệnh


def test_cli_loss_writes_every_output(tmp_path: Path):
    src = tmp_path / "log.csv"
    _history().to_csv(src, index=False)
    png, pdf = tmp_path / "loss.png", tmp_path / "out" / "loss.pdf"
    assert main(["loss", str(src), "-o", str(png), "-o", str(pdf), "--smooth", "0.8", "--log-y"]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert pdf.read_bytes()[:5] == b"%PDF-"


def test_cli_compare(tmp_path: Path):
    src = tmp_path / "scores.csv"
    _scores().to_csv(src, index=False)
    out = tmp_path / "compare.svg"
    assert main(["compare", str(src), "-o", str(out), "--metrics", "cer", "--order", "E1", "baseline"]) == 0
    assert "<svg" in out.read_text(encoding="utf-8")


def test_cli_bad_input_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    src = tmp_path / "scores.csv"
    _scores().assign(metric="mos").to_csv(src, index=False)
    assert main(["compare", str(src), "-o", str(tmp_path / "x.png")]) == 2
    assert "mos" in capsys.readouterr().err
