# """
# paper_plots.py — generate EMNLP-style paper figures from eval.py output files.

# Usage
# -----
# python paper_plots.py --inputs results/*_eval.json --output-dir figures/
# python paper_plots.py \\
#     --inputs results/collisions_*_eval.json \\
#     --output-dir figures/ --task collisions
# """

# import argparse
# import glob
# import json
# import os
# from collections import defaultdict

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np

# # ---------------------------------------------------------------------------
# # EMNLP / ACL style constants
# # ---------------------------------------------------------------------------

# COLORS = [
#     "#2166AC",   # blue      — instruct
#     "#D6604D",   # red       — think
#     "#4DAC26",   # green     — hybrid instruct
#     "#E08B00",   # amber     — hybrid think
#     "#762A83",   # purple
#     "#1B7837",   # dark green
#     "#BF812D",   # brown
#     "#35978F",   # teal
# ]

# MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# RANDOM_BASELINE  = {4: 0.25}
# DEFAULT_BASELINE = 0.25
# DYCK_BASELINE    = 0.33

# plt.rcParams.update({
#     # --- typography (ACL / EMNLP papers use Times) ---
#     "font.family":        "serif",
#     "font.serif":         ["Times New Roman", "DejaVu Serif"],
#     "font.size":          13,
#     "axes.labelsize":     14,
#     "axes.titlesize":     14,
#     "legend.fontsize":    8.5,
#     "xtick.labelsize":    11,
#     "ytick.labelsize":    11,
#     # --- clean frame ---
#     "axes.spines.top":    False,
#     "axes.spines.right":  False,
#     # --- grid ---
#     "axes.grid":          True,
#     "grid.linestyle":     "--",
#     "grid.alpha":         0.4,
#     "grid.color":         "#cccccc",
#     # --- output ---
#     "figure.dpi":         150,
#     "savefig.dpi":        300,
#     "savefig.bbox":       "tight",
# })

# # Shared legend kwargs — compact, no long line handles
# LEGEND_KW = dict(
#     framealpha=0.9,
#     edgecolor="#cccccc",
#     ncol=1,
#     handlelength=0.8,
#     handletextpad=0.3,
#     borderpad=0.4,
#     labelspacing=0.25,
#     markerscale=0.65,
# )

# # ---------------------------------------------------------------------------
# # Data loading  (unchanged from original)
# # ---------------------------------------------------------------------------

# def load_eval_file(path: str) -> dict:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def collect_series(eval_files, max_m=None):
#     raw = {}
#     tasks_found = set()

#     for path in sorted(eval_files):
#         data  = load_eval_file(path)
#         model = data.get("model_name", os.path.splitext(os.path.basename(path))[0])
#         short = model.split("/")[-1] if "/" in model else model
#         samples = data.get("scored_samples") or data.get("samples") or []
#         if samples:
#             shot = samples[0].get("shot", "zero")
#         else:
#             fname = os.path.basename(path)
#             shot  = "one" if "oneshot" in fname else "zero"
#         short = f"{short} ({shot})"
#         task  = data.get("task", "unknown")
#         tasks_found.add(task)

#         if short not in raw:
#             raw[short] = {"accuracy": {}, "pwa": {}, "parse_rate": {}}

#         for key, d in data.get("per_m_n", {}).items():
#             try:
#                 m_str, n_str = key.split("x")
#                 m, n = int(m_str), int(n_str)
#             except ValueError:
#                 continue
#             if max_m is not None and m > max_m:
#                 continue
#             pair = (m, n)

#             if d.get("accuracy") is not None:
#                 raw[short]["accuracy"].setdefault(pair, []).append(d["accuracy"])
#             if d.get("parsed_weighted_accuracy") is not None:
#                 raw[short]["pwa"].setdefault(pair, []).append(d["parsed_weighted_accuracy"])
#             if d.get("n_scored") is not None and d.get("n_total", 0) > 0:
#                 raw[short]["parse_rate"].setdefault(pair, []).append(
#                     d["n_scored"] / d["n_total"]
#                 )

#     series = {}
#     for short, metrics in raw.items():
#         acc_mean, acc_std   = {}, {}
#         pwa_mean, pwa_std   = {}, {}
#         rate_mean, rate_std = {}, {}

#         for pair, vals in metrics["accuracy"].items():
#             acc_mean[pair] = float(np.mean(vals))
#             acc_std[pair]  = float(np.std(vals)) if len(vals) > 1 else 0.0

#         for pair, vals in metrics["pwa"].items():
#             pwa_mean[pair] = float(np.mean(vals))
#             pwa_std[pair]  = float(np.std(vals)) if len(vals) > 1 else 0.0

#         for pair, vals in metrics["parse_rate"].items():
#             rate_mean[pair] = float(np.mean(vals))
#             rate_std[pair]  = float(np.std(vals)) if len(vals) > 1 else 0.0

#         if acc_mean:
#             series[short] = {
#                 "accuracy":       acc_mean,
#                 "accuracy_std":   acc_std,
#                 "pwa":            pwa_mean,
#                 "pwa_std":        pwa_std,
#                 "parse_rate":     rate_mean,
#                 "parse_rate_std": rate_std,
#             }

#     return series, tasks_found


# # ---------------------------------------------------------------------------
# # Shared helper
# # ---------------------------------------------------------------------------

# def _annotate_values(ax, xs, ys, color):
#     """Print value labels above each data point."""
#     for x, y in zip(xs, ys):
#         if not np.isnan(y):
#             ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 7),
#                         textcoords="offset points",
#                         ha="center", va="bottom",
#                         fontsize=8.5, color=color, fontweight="bold")


# def _save(fig, output_dir, stem):
#     for ext in ("pdf", "png"):
#         p = os.path.join(output_dir, f"{stem}.{ext}")
#         fig.savefig(p)
#         print(f"  Saved → {p}")
#     plt.close(fig)


# # ---------------------------------------------------------------------------
# # Plot 1 — Accuracy line
# # ---------------------------------------------------------------------------

# def plot_line(series, output_dir, title="", task=None):
#     all_pairs = sorted({pair for s in series.values() for pair in s["accuracy"]})
#     x_labels  = [f"({m},{n})" for m, n in all_pairs]
#     xs        = list(range(len(all_pairs)))

#     fig, ax = plt.subplots(figsize=(max(9, len(all_pairs) * 0.8), 4.8))

#     baseline = DYCK_BASELINE if task and "dyck" in task else DEFAULT_BASELINE
#     ax.axhline(baseline, color="#888888", linestyle="--", linewidth=1.2,
#                label=f"Random baseline ({baseline:.2f})", zorder=1)

#     for i, (model, s) in enumerate(sorted(series.items())):
#         color  = COLORS[i % len(COLORS)]
#         marker = MARKERS[i % len(MARKERS)]
#         ys     = [s["accuracy"].get(pair, float("nan")) for pair in all_pairs]
#         ys_std = [s["accuracy_std"].get(pair, 0.0)      for pair in all_pairs]

#         ax.plot(xs, ys, label=model, color=color,
#                 marker=marker, linewidth=1.8, markersize=7, zorder=3)

#         ys_arr  = np.array(ys,     dtype=float)
#         std_arr = np.array(ys_std, dtype=float)
#         if std_arr.any():
#             ax.fill_between(xs,
#                             np.clip(ys_arr - std_arr, 0, 1),
#                             np.clip(ys_arr + std_arr, 0, 1),
#                             alpha=0.15, color=color)

#         _annotate_values(ax, xs, ys, color)

#     ax.set_xticks(xs)
#     ax.set_xticklabels(x_labels, rotation=0)
#     ax.set_xlabel("(m, n)")
#     ax.set_ylabel("Accuracy")
#     ax.set_ylim(-0.05, 1.15)
#     ax.legend(loc="lower left", **LEGEND_KW)
#     # title intentionally omitted for paper style

#     plt.tight_layout()
#     task_str    = f"{task}_" if task else ""
#     model_names = "_".join(sorted(series.keys()))
#     _save(fig, output_dir, f"accuracy_line_{task_str}{model_names}")


# # ---------------------------------------------------------------------------
# # Plot 2 — Parsed Weighted Accuracy line
# # ---------------------------------------------------------------------------

# def plot_pwa_line(series, output_dir, title="", task=None):
#     all_pairs = sorted({pair for s in series.values() for pair in s["pwa"]})
#     x_labels  = [f"({m},{n})" for m, n in all_pairs]
#     xs        = list(range(len(all_pairs)))

#     fig, ax = plt.subplots(figsize=(max(9, len(all_pairs) * 0.8), 4.8))

#     baseline = DYCK_BASELINE if task and "dyck" in task else DEFAULT_BASELINE
#     ax.axhline(baseline, color="#888888", linestyle="--", linewidth=1.2,
#                label=f"Random baseline ({baseline:.2f})", zorder=1)

#     for i, (model, s) in enumerate(sorted(series.items())):
#         color  = COLORS[i % len(COLORS)]
#         marker = MARKERS[i % len(MARKERS)]
#         ys     = [s["pwa"].get(pair, float("nan")) for pair in all_pairs]
#         ys_std = [s["pwa_std"].get(pair, 0.0)      for pair in all_pairs]

#         ax.plot(xs, ys, label=model, color=color,
#                 marker=marker, linewidth=1.8, markersize=7, zorder=3)

#         ys_arr  = np.array(ys,     dtype=float)
#         std_arr = np.array(ys_std, dtype=float)
#         if std_arr.any():
#             ax.fill_between(xs,
#                             np.clip(ys_arr - std_arr, 0, 1),
#                             np.clip(ys_arr + std_arr, 0, 1),
#                             alpha=0.15, color=color)

#         _annotate_values(ax, xs, ys, color)

#     ax.set_xticks(xs)
#     ax.set_xticklabels(x_labels, rotation=0)
#     ax.set_xlabel("(m, n)")
#     ax.set_ylabel("Parsed Weighted Accuracy")
#     ax.set_ylim(-0.05, 1.15)
#     ax.legend(loc="lower left", **LEGEND_KW)

#     plt.tight_layout()
#     task_str    = f"{task}_" if task else ""
#     model_names = "_".join(sorted(series.keys()))
#     _save(fig, output_dir, f"pwa_line_{task_str}{model_names}")


# # ---------------------------------------------------------------------------
# # Plot 3 — Parse rate line
# # ---------------------------------------------------------------------------

# def plot_parse_rate(series, output_dir, title="", task=None):
#     all_pairs = sorted({pair for s in series.values() for pair in s["parse_rate"]})
#     x_labels  = [f"({m},{n})" for m, n in all_pairs]
#     xs        = list(range(len(all_pairs)))

#     fig, ax = plt.subplots(figsize=(max(9, len(all_pairs) * 0.8), 4.8))

#     ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.2,
#                label="Perfect parse rate (1.0)", zorder=1)

#     for i, (model, s) in enumerate(sorted(series.items())):
#         color  = COLORS[i % len(COLORS)]
#         marker = MARKERS[i % len(MARKERS)]
#         ys     = [s["parse_rate"].get(pair, float("nan")) for pair in all_pairs]
#         ys_std = [s["parse_rate_std"].get(pair, 0.0)      for pair in all_pairs]

#         ax.plot(xs, ys, label=model, color=color,
#                 marker=marker, linewidth=1.8, markersize=7, zorder=3)

#         ys_arr  = np.array(ys,     dtype=float)
#         std_arr = np.array(ys_std, dtype=float)
#         if std_arr.any():
#             ax.fill_between(xs,
#                             np.clip(ys_arr - std_arr, 0, 1),
#                             np.clip(ys_arr + std_arr, 0, 1),
#                             alpha=0.15, color=color)

#         _annotate_values(ax, xs, ys, color)

#     ax.set_xticks(xs)
#     ax.set_xticklabels(x_labels, rotation=0)
#     ax.set_xlabel("(m, n)")
#     ax.set_ylabel("Parse Rate (n_scored / n_total)")
#     ax.set_ylim(-0.05, 1.15)
#     ax.legend(loc="lower left", **LEGEND_KW)

#     plt.tight_layout()
#     task_str    = f"{task}_" if task else ""
#     model_names = "_".join(sorted(series.keys()))
#     _save(fig, output_dir, f"parse_rate_line_{task_str}{model_names}")


# # ---------------------------------------------------------------------------
# # CLI  (unchanged from original)
# # ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate EMNLP-style paper figures from eval.py output files."
#     )
#     parser.add_argument("--inputs", nargs="+", required=True,
#                         help="Paths (or globs) to eval JSON files.")
#     parser.add_argument("--output-dir", default="figures",
#                         help="Directory to write figures (default: figures/).")
#     parser.add_argument("--title", default="",
#                         help="Optional figure title suffix (unused in EMNLP style).")
#     parser.add_argument("--no-heatmap", action="store_true",
#                         help="Skip the heatmap (kept for CLI compatibility).")
#     parser.add_argument("--max-m", type=int, default=None,
#                         help="Cap plots at this m value.")
#     parser.add_argument("--task", default=None,
#                         help="Task name to include in output filenames.")
#     args = parser.parse_args()

#     eval_files = []
#     for pat in args.inputs:
#         matches = glob.glob(pat)
#         eval_files.extend(matches if matches else [pat])
#     eval_files = sorted(set(eval_files))

#     if not eval_files:
#         print("No eval files found.", file=__import__("sys").stderr)
#         raise SystemExit(1)

#     print(f"Loading {len(eval_files)} eval file(s) …")
#     for f in eval_files:
#         print(f"  {f}")

#     series, tasks_found = collect_series(eval_files, max_m=args.max_m)
#     if not series:
#         print("No plottable data found.")
#         return

#     task_str = args.task or ("_".join(sorted(tasks_found)) if tasks_found else "unknown")

#     os.makedirs(args.output_dir, exist_ok=True)
#     print(f"\nModels : {sorted(series.keys())}")
#     print(f"Task   : {task_str}")
#     print(f"Output : {args.output_dir}\n")

#     plot_line(series,       args.output_dir, title=args.title, task=task_str)
#     plot_pwa_line(series,   args.output_dir, title=args.title, task=task_str)
#     plot_parse_rate(series, args.output_dir, title=args.title, task=task_str)

#     print("\nAll figures written.")


# if __name__ == "__main__":
#     main()


"""
paper_plots.py — generate paper-quality figures from eval.py output files,
styled to match plot_template.py (serif fonts, clean spines, dotted y-grid,
above-axes legend, per-point value annotations).

Usage
-----
python paper_plots.py --inputs results/*_eval.json --output-dir figures/
python paper_plots.py \\
    --inputs results/collisions_*_eval.json \\
    --output-dir figures/ --task collisions
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Style constants  (from plot_template.py)
# ---------------------------------------------------------------------------

GRID_COLOR       = "#9CA3AF"
TEXT_COLOR       = "#111827"
MUTED_TEXT_COLOR = "#4B5563"

LINE_COLORS = [
    "#0072B2",  # blue      — instruct
    "#E06060",  # red       — think
    "#009E73",  # green     — hybrid instruct
    "#E69F00",  # amber     — hybrid think
    "#CC79A7",  # magenta
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update(
    {
        # Typography
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        # Sizes
        "font.size":          8,
        "axes.titlesize":     9,
        "axes.labelsize":     8,
        "xtick.labelsize":    7.2,
        "ytick.labelsize":    7.2,
        "legend.fontsize":    7.0,
        # Axes
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.edgecolor":     MUTED_TEXT_COLOR,
        # Ticks
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        # Export
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        # Background
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    }
)

# Baseline config
DEFAULT_BASELINE = 0.25
DYCK_BASELINE    = 0.33

# ---------------------------------------------------------------------------
# Shared styling helpers  (from plot_template.py)
# ---------------------------------------------------------------------------

def _style_axes(ax, xlabel=None, ylabel=None):
    """Apply plot_template.py axis style: dotted y-grid, clean spines, labels."""
    ax.grid(
        True, axis="y",
        linestyle=":", linewidth=0.45,
        color=GRID_COLOR, alpha=0.65,
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color(MUTED_TEXT_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _save_figure(fig, output_path):
    """Save as both PDF and PNG (300 dpi), matching plot_template.py."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(output_path.with_suffix(".png"),
                dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print(f"  Saved -> {output_path.with_suffix('.pdf')}")
    print(f"  Saved -> {output_path.with_suffix('.png')}")
    plt.close(fig)


def _annotate_all(ax, xs, ys, color):
    """Print bold value labels above every data point."""
    for x, y in zip(xs, ys):
        if not np.isnan(y):
            ax.annotate(
                f"{y:.2f}",
                xy=(x, y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=5.5, fontweight="bold",
                color=color,
            )


def _above_legend(ax, handles, ncol=2):
    """Place a frameless legend above the axes, matching plot_template.py."""
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
        fontsize=6.2,
    )


def _ylim_from_data(ax, all_ys, hline=None, padding=0.18):
    """Set y-limits with padding, matching plot_template.py logic."""
    flat = np.concatenate([np.asarray(y) for y in all_ys])
    if hline is not None:
        flat = np.append(flat, hline)
    y_min = np.nanmin(flat)
    y_max = np.nanmax(flat)
    pad = max((y_max - y_min) * padding, 0.03)
    ax.set_ylim(y_min - pad, y_max + pad * 2.5)


# ---------------------------------------------------------------------------
# Data loading  (unchanged)
# ---------------------------------------------------------------------------

def load_eval_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_series(eval_files, max_m=None):
    raw = {}
    tasks_found = set()

    for path in sorted(eval_files):
        data  = load_eval_file(path)
        model = data.get("model_name", os.path.splitext(os.path.basename(path))[0])
        short = model.split("/")[-1] if "/" in model else model
        samples = data.get("scored_samples") or data.get("samples") or []
        if samples:
            shot = samples[0].get("shot", "zero")
        else:
            fname = os.path.basename(path)
            shot  = "one" if "oneshot" in fname else "zero"
        short = f"{short} ({shot})"
        task  = data.get("task", "unknown")
        tasks_found.add(task)

        if short not in raw:
            raw[short] = {"accuracy": {}, "pwa": {}, "parse_rate": {}}

        for key, d in data.get("per_m_n", {}).items():
            try:
                m_str, n_str = key.split("x")
                m, n = int(m_str), int(n_str)
            except ValueError:
                continue
            if max_m is not None and m > max_m:
                continue
            pair = (m, n)

            if d.get("accuracy") is not None:
                raw[short]["accuracy"].setdefault(pair, []).append(d["accuracy"])
            if d.get("parsed_weighted_accuracy") is not None:
                raw[short]["pwa"].setdefault(pair, []).append(d["parsed_weighted_accuracy"])
            if d.get("n_scored") is not None and d.get("n_total", 0) > 0:
                raw[short]["parse_rate"].setdefault(pair, []).append(
                    d["n_scored"] / d["n_total"]
                )

    series = {}
    for short, metrics in raw.items():
        acc_mean, acc_std   = {}, {}
        pwa_mean, pwa_std   = {}, {}
        rate_mean, rate_std = {}, {}

        for pair, vals in metrics["accuracy"].items():
            acc_mean[pair] = float(np.mean(vals))
            acc_std[pair]  = float(np.std(vals)) if len(vals) > 1 else 0.0

        for pair, vals in metrics["pwa"].items():
            pwa_mean[pair] = float(np.mean(vals))
            pwa_std[pair]  = float(np.std(vals)) if len(vals) > 1 else 0.0

        for pair, vals in metrics["parse_rate"].items():
            rate_mean[pair] = float(np.mean(vals))
            rate_std[pair]  = float(np.std(vals)) if len(vals) > 1 else 0.0

        if acc_mean:
            series[short] = {
                "accuracy":       acc_mean,
                "accuracy_std":   acc_std,
                "pwa":            pwa_mean,
                "pwa_std":        pwa_std,
                "parse_rate":     rate_mean,
                "parse_rate_std": rate_std,
            }

    return series, tasks_found


# ---------------------------------------------------------------------------
# Core line-plot builder
# ---------------------------------------------------------------------------

def _line_plot(series, metric_key, std_key, output_dir, stem,
               ylabel, hline_val, hline_label, figsize=None):
    """
    Generic styled line plot matching plot_template.py layout:
      - above-axes 2-col frameless legend
      - dotted y-only grid
      - per-point bold value annotations
      - clean left+bottom spines only
      - no title
    """
    all_pairs = sorted({pair for s in series.values() for pair in s[metric_key]})
    x_labels  = [f"({m},{n})" for m, n in all_pairs]
    xs        = list(range(len(all_pairs)))

    n = len(all_pairs)
    if figsize is None:
        figsize = (max(4.5, n * 0.75), 2.8)

    fig, ax = plt.subplots(figsize=figsize, dpi=150, constrained_layout=True)

    # Reference horizontal line
    ax.axhline(hline_val, color="#888888", linewidth=0.9,
               linestyle="--", zorder=1)

    all_ys = []
    legend_handles = [
        mlines.Line2D([], [], color="#888888", linewidth=0.9,
                      linestyle="--", label=hline_label)
    ]

    for i, (model, s) in enumerate(sorted(series.items())):
        color  = LINE_COLORS[i % len(LINE_COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ys     = [s[metric_key].get(pair, float("nan")) for pair in all_pairs]
        ys_std = [s[std_key].get(pair, 0.0)             for pair in all_pairs]

        ax.plot(xs, ys, color=color, marker=marker,
                linewidth=1.15, markersize=3.5,
                markeredgewidth=0, alpha=0.95, zorder=3)

        # Optional shaded std band
        ys_arr  = np.array(ys,     dtype=float)
        std_arr = np.array(ys_std, dtype=float)
        if std_arr.any():
            ax.fill_between(
                xs,
                np.clip(ys_arr - std_arr, 0, 1),
                np.clip(ys_arr + std_arr, 0, 1),
                alpha=0.12, color=color, zorder=2,
            )

        _annotate_all(ax, xs, ys, color)
        all_ys.append(ys)

        legend_handles.append(
            mlines.Line2D([], [], color=color, linewidth=1.15,
                          marker=marker, markersize=3.5,
                          markeredgewidth=0, label=model)
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=6.8)

    _ylim_from_data(ax, all_ys, hline=hline_val)
    _style_axes(ax, xlabel="(m, n)", ylabel=ylabel)
    _above_legend(ax, legend_handles, ncol=2)

    _save_figure(fig, os.path.join(output_dir, stem))


# ---------------------------------------------------------------------------
# Public plot functions  (drop-in replacements for originals)
# ---------------------------------------------------------------------------

def plot_line(series, output_dir, title="", task=None):
    baseline    = DYCK_BASELINE if task and "dyck" in task else DEFAULT_BASELINE
    task_str    = f"{task}_" if task else ""
    model_names = "_".join(sorted(series.keys()))
    _line_plot(
        series,
        metric_key  = "accuracy",
        std_key     = "accuracy_std",
        output_dir  = output_dir,
        stem        = f"accuracy_line_{task_str}{model_names}",
        ylabel      = "Accuracy",
        hline_val   = baseline,
        hline_label = f"Random baseline ({baseline:.2f})",
    )


def plot_pwa_line(series, output_dir, title="", task=None):
    baseline    = DYCK_BASELINE if task and "dyck" in task else DEFAULT_BASELINE
    task_str    = f"{task}_" if task else ""
    model_names = "_".join(sorted(series.keys()))
    _line_plot(
        series,
        metric_key  = "pwa",
        std_key     = "pwa_std",
        output_dir  = output_dir,
        stem        = f"pwa_line_{task_str}{model_names}",
        ylabel      = "Parsed Weighted Accuracy",
        hline_val   = baseline,
        hline_label = f"Random baseline ({baseline:.2f})",
    )


def plot_parse_rate(series, output_dir, title="", task=None):
    task_str    = f"{task}_" if task else ""
    model_names = "_".join(sorted(series.keys()))
    _line_plot(
        series,
        metric_key  = "parse_rate",
        std_key     = "parse_rate_std",
        output_dir  = output_dir,
        stem        = f"parse_rate_line_{task_str}{model_names}",
        ylabel      = "Parse Rate (n_scored / n_total)",
        hline_val   = 1.0,
        hline_label = "Perfect parse rate (1.0)",
    )


# ---------------------------------------------------------------------------
# CLI  (unchanged from original)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from eval.py output files."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths (or globs) to eval JSON files.")
    parser.add_argument("--output-dir", default="figures",
                        help="Directory to write figures (default: figures/).")
    parser.add_argument("--title", default="",
                        help="Optional figure title suffix (unused in paper style).")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip the heatmap (kept for CLI compatibility).")
    parser.add_argument("--max-m", type=int, default=None,
                        help="Cap plots at this m value.")
    parser.add_argument("--task", default=None,
                        help="Task name to include in output filenames.")
    args = parser.parse_args()

    eval_files = []
    for pat in args.inputs:
        matches = glob.glob(pat)
        eval_files.extend(matches if matches else [pat])
    eval_files = sorted(set(eval_files))

    if not eval_files:
        print("No eval files found.", file=__import__("sys").stderr)
        raise SystemExit(1)

    print(f"Loading {len(eval_files)} eval file(s) ...")
    for f in eval_files:
        print(f"  {f}")

    series, tasks_found = collect_series(eval_files, max_m=args.max_m)
    if not series:
        print("No plottable data found.")
        return

    task_str = args.task or ("_".join(sorted(tasks_found)) if tasks_found else "unknown")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nModels : {sorted(series.keys())}")
    print(f"Task   : {task_str}")
    print(f"Output : {args.output_dir}\n")

    plot_line(series,       args.output_dir, title=args.title, task=task_str)
    plot_pwa_line(series,   args.output_dir, title=args.title, task=task_str)
    plot_parse_rate(series, args.output_dir, title=args.title, task=task_str)

    print("\nAll figures written.")


if __name__ == "__main__":
    main()