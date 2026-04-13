#!/usr/bin/env python3
"""
Build matplotlib figures for the course report.

Reads optional training metric CSVs (from ``--metrics_csv``) and a merged evaluation
summary JSON (from ``scripts/build_eval_summary.py``). The efficiency figure uses
parameters in **thousands** (×10³) so small students (e.g. ``gaze_micro`` ~50k) stay
visible next to a MobileNetV3-Small teacher (~1.5M). A separate **predict_speed.pdf**
compares ms/image (and approximate FPS). ``teacher_arch`` / ``student_arch`` from
``evaluate.py`` JSON appear in subtitles when present. Loss curves from CSVs are
trimmed to the first **20 epochs** by default (``--max_plot_epochs``) so long KD runs do not
crowd the x-axis; use ``--max_plot_epochs 0`` to plot every logged epoch.

Usage (from ``gaze_kd_project``)::

    python scripts/make_paper_figures.py --summary runs/summary.json \\
        --metrics_teacher runs/m_teacher.csv --metrics_student runs/m_student.csv \\
        --metrics_kd runs/m_kd.csv --out_dir paper/figures

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_supervised_csv(path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs, train_mse, val_mse = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_mse.append(float(row["train_mse"]))
            val_mse.append(float(row["val_mse"]))
    return epochs, train_mse, val_mse


def read_kd_csv(path: Path) -> tuple[list[int], list[float]]:
    epochs, val_gt = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            val_gt.append(float(row["val_mse_gt"]))
    return epochs, val_gt


def _human_param_count(n: float) -> str:
    """Format parameter count for bar annotations (n is raw count from evaluate JSON)."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return str(int(n))


def _arch_caption(summary: dict) -> str:
    """One-line caption from evaluate.py JSON fields (teacher_arch / student_arch)."""
    if not summary:
        return ""
    parts: list[str] = []
    te = summary.get("teacher") or {}
    ta = te.get("teacher_arch")
    if isinstance(ta, str) and ta.strip():
        parts.append(f"teacher={ta.strip()}")
    sb = summary.get("student_baseline") or {}
    sk = summary.get("student_kd") or {}
    sa = sb.get("student_arch") or sk.get("student_arch")
    if isinstance(sa, str) and sa.strip():
        parts.append(f"student={sa.strip()}")
    return " · ".join(parts)


def _fps_from_entry(entry: dict | None, ms: float) -> float:
    if entry and entry.get("fps") is not None:
        return float(entry["fps"])
    return (1000.0 / ms) if ms > 0 else 0.0


def _trim_series_head(
    max_epochs: int, epochs: list[int], *series: list[float]
) -> tuple[list[int], ...]:
    """Keep only the first ``max_epochs`` rows (same index range for all series)."""
    if max_epochs <= 0 or len(epochs) <= max_epochs:
        return (epochs,) + tuple(series)
    return (epochs[:max_epochs],) + tuple(s[:max_epochs] for s in series)


def plot_loss_curves(
    out_path: Path,
    teacher: Path | None,
    student: Path | None,
    kd: Path | None,
    demo: bool,
    max_plot_epochs: int = 20,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    if demo:
        t = np.arange(1, 21)
        ax.plot(t, 0.35 * np.exp(-0.18 * t) + 0.02, label="Teacher val MSE")
        ax.plot(t, 0.48 * np.exp(-0.14 * t) + 0.045, label="Student baseline val MSE")
        ax.plot(t, 0.42 * np.exp(-0.15 * t) + 0.028, label="Student + KD val MSE (GT term)")
        ax.set_title("Validation error vs. epoch (example — use real CSVs after training)")
    else:
        assert teacher is not None and student is not None and kd is not None
        e1, _tr1, v1 = read_supervised_csv(teacher)
        e2, _tr2, v2 = read_supervised_csv(student)
        e3, v3 = read_kd_csv(kd)
        if max_plot_epochs > 0:
            e1, _, v1 = _trim_series_head(max_plot_epochs, e1, _tr1, v1)
            e2, _, v2 = _trim_series_head(max_plot_epochs, e2, _tr2, v2)
            e3, v3 = _trim_series_head(max_plot_epochs, e3, v3)
        ax.plot(e1, v1, marker="o", markersize=3, label="Teacher val MSE")
        ax.plot(e2, v2, marker="o", markersize=3, label="Student baseline val MSE")
        ax.plot(e3, v3, marker="o", markersize=3, label="Student + KD (val MSE vs. GT)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_metric_bars(summary: dict, out_path: Path, demo: bool) -> None:
    labels = ["Teacher", "Student\n(baseline)", "Student\n+ KD"]
    if demo or not summary:
        mse = [0.031, 0.052, 0.041]
        mae = [0.112, 0.148, 0.125]
        l2 = [0.145, 0.189, 0.162]
    else:
        mse = [
            summary["teacher"]["mse"],
            summary["student_baseline"]["mse"],
            summary["student_kd"]["mse"],
        ]
        mae = [
            summary["teacher"]["mae"],
            summary["student_baseline"]["mae"],
            summary["student_kd"]["mae"],
        ]
        l2 = [
            summary["teacher"]["mean_l2"],
            summary["student_baseline"]["mean_l2"],
            summary["student_kd"]["mean_l2"],
        ]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w, mse, width=w, label="MSE")
    ax.bar(x, mae, width=w, label="MAE")
    ax.bar(x + w, l2, width=w, label="Mean L2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Error (label units)")
    ax.set_title("Test/val metrics by model")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    cap = _arch_caption(summary)
    if cap:
        fig.suptitle(cap, fontsize=8, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_efficiency(summary: dict, out_path: Path, demo: bool) -> None:
    # Use params in thousands so MV3-Small (~1.5M) and gaze_micro (~0.1M) are both visible.
    if demo or not summary:
        names = ["Teacher", "Student\n(baseline)", "Student\n+ KD"]
        params_k = [1520.0, 49.4, 49.4]
        params_raw = [1.52e6, 49_426, 49_426]
        ms = [1.25, 0.28, 0.29]
    else:
        names = ["Teacher", "Student\n(baseline)", "Student\n+ KD"]
        p0 = float(summary["teacher"]["params"])
        p1 = float(summary["student_baseline"]["params"])
        p2 = float(summary["student_kd"]["params"])
        params_raw = [p0, p1, p2]
        params_k = [p / 1e3 for p in params_raw]
        ms = [
            summary["teacher"]["ms_per_image"],
            summary["student_baseline"]["ms_per_image"],
            summary["student_kd"]["ms_per_image"],
        ]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars0 = axes[0].bar(names, params_k, color=colors)
    axes[0].set_ylabel("Parameters (×10³)")
    axes[0].set_title("Model size")
    axes[0].grid(True, axis="y", alpha=0.3)
    ymax0 = max(params_k) * 1.18 if params_k else 1.0
    axes[0].set_ylim(0, ymax0)
    for bar, pr in zip(bars0, params_raw):
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02 * ymax0,
            _human_param_count(pr),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    bars1 = axes[1].bar(names, ms, color=colors)
    axes[1].set_ylabel("ms / image")
    axes[1].set_title("Latency (dummy input, batch=1)")
    axes[1].grid(True, axis="y", alpha=0.3)
    ymax1 = max(ms) * 1.2 if ms else 1.0
    for bar in bars1:
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02 * ymax1,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    cap = _arch_caption(summary) if not demo and summary else ""
    fig.suptitle("Efficiency comparison" + (f" — {cap}" if cap else ""), fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_predict_speed(summary: dict, out_path: Path, demo: bool) -> None:
    """Bar chart of ms/image with FPS annotations (from JSON ``fps`` or 1000/ms)."""
    names = ["Teacher", "Student\n(baseline)", "Student\n+ KD"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    keys = ("teacher", "student_baseline", "student_kd")

    if demo or not summary:
        ms = [1.25, 0.28, 0.29]
        entries: tuple[dict | None, dict | None, dict | None] = (None, None, None)
    else:
        ms = [float(summary[k]["ms_per_image"]) for k in keys]
        entries = tuple(summary[k] for k in keys)

    fps_list = [_fps_from_entry(e, m) for e, m in zip(entries, ms)]

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    bars = ax.bar(names, ms, color=colors)
    ax.set_ylabel("ms / image")
    ax.set_title("Prediction speed (lower ms = faster; dummy 224×224, batch = 1)")
    ax.grid(True, axis="y", alpha=0.3)
    ymax = max(ms) * 1.3 if ms else 1.0
    ax.set_ylim(0, ymax)
    for bar, m, f in zip(bars, ms, fps_list):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02 * ymax,
            f"{m:.2f} ms\n≈ {f:.0f} FPS",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    cap = _arch_caption(summary) if not demo and summary else ""
    fig.suptitle("Inference speed comparison" + (f" — {cap}" if cap else ""), fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_scatter_npz(npz_path: Path, out_path: Path) -> None:
    data = np.load(npz_path)
    pred = data["pred"]
    gt = data["gt"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for i, name in enumerate(["gaze_x", "gaze_y"]):
        axes[i].scatter(gt[:, i], pred[:, i], s=8, alpha=0.45, edgecolors="none")
        lims = [min(gt[:, i].min(), pred[:, i].min()), max(gt[:, i].max(), pred[:, i].max())]
        axes[i].plot(lims, lims, "k--", linewidth=1)
        axes[i].set_xlabel(f"GT {name}")
        axes[i].set_ylabel(f"Pred {name}")
        axes[i].set_title(name)
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Predictions vs. ground truth (student + KD)")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def load_summary(path: Path | None) -> dict:
    if not path or not path.is_file():
        return {}
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper figures")
    p.add_argument("--out_dir", type=str, default="paper/figures")
    p.add_argument("--summary", type=str, default="", help="merged eval JSON")
    p.add_argument("--metrics_teacher", type=str, default="")
    p.add_argument("--metrics_student", type=str, default="")
    p.add_argument("--metrics_kd", type=str, default="")
    p.add_argument("--scatter_npz", type=str, default="", help="optional predictions from evaluate.py")
    p.add_argument("--demo", action="store_true", help="use placeholder data when files missing")
    p.add_argument(
        "--max_plot_epochs",
        type=int,
        default=20,
        help="loss_curves.pdf: use only the first N rows from each metrics CSV (default 20); 0 = no trim",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary) if args.summary else None
    summary = load_summary(summary_path)
    demo = args.demo or not summary

    mt = Path(args.metrics_teacher) if args.metrics_teacher else None
    ms = Path(args.metrics_student) if args.metrics_student else None
    mk = Path(args.metrics_kd) if args.metrics_kd else None

    have_curve_csvs = bool(
        mt and ms and mk and mt.is_file() and ms.is_file() and mk.is_file()
    )
    use_demo_curves = args.demo or not have_curve_csvs
    if use_demo_curves:
        print(
            "loss_curves.pdf: using synthetic decay curves (not your training logs). "
            "To plot real val MSE: train with --metrics_csv on teacher, student, and KD, "
            "then pass --metrics_teacher, --metrics_student, --metrics_kd to this script "
            "(and do not use --demo)."
        )
    else:
        msg = f"loss_curves.pdf: from CSVs {mt} {ms} {mk}"
        if args.max_plot_epochs > 0:
            msg += f" (first {args.max_plot_epochs} epochs per series)"
        print(msg)
    plot_loss_curves(
        out_dir / "loss_curves.pdf",
        mt,
        ms,
        mk,
        demo=use_demo_curves,
        max_plot_epochs=args.max_plot_epochs,
    )

    # Fix demo logic for bars: demo flag OR incomplete summary
    bar_demo = args.demo or not {"teacher", "student_baseline", "student_kd"}.issubset(summary.keys())
    plot_metric_bars(summary if not bar_demo else {}, out_dir / "metrics_compare.pdf", demo=bar_demo)
    plot_efficiency(summary if not bar_demo else {}, out_dir / "efficiency.pdf", demo=bar_demo)
    plot_predict_speed(summary if not bar_demo else {}, out_dir / "predict_speed.pdf", demo=bar_demo)
    print("Wrote predict_speed.pdf")

    if args.scatter_npz and Path(args.scatter_npz).is_file():
        plot_scatter_npz(Path(args.scatter_npz), out_dir / "gaze_scatter.pdf")
        print("Wrote gaze_scatter.pdf")
    else:
        print("Skipping gaze scatter (provide --scatter_npz pointing to a .npz)")

    print(f"Wrote figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
