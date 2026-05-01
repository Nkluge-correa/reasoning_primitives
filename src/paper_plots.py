"""
paper_plots.py — generate paper-quality figures from eval.py output files.

Usage
-----
# Single eval file
python paper_plots.py --inputs results/*_eval.json --output-dir figures/

# Or compare across tasks
python paper_plots.py \\
    --inputs results/collisions_*_eval.json \\
    --group-by task \\
    --output-dir figures/

Figures produced
----------------
1. accuracy_vs_difficulty_line.pdf  — line plot, one curve per model
2. accuracy_vs_difficulty_bar.pdf   — grouped bar chart
3. accuracy_heatmap.pdf             — heatmap (models × difficulties)

All figures are also saved as .png (300 dpi) for quick preview.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

COLORS  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#FF5722", "#795548", "#00ACC1"]
MARKERS = ["o",        "s",       "^",       "D",       "v",       "P",       "X",       "*"]

RANDOM_BASELINE = {4: 0.25}          # 4-option MC -> 0.25; 2-option -> 0.50
DEFAULT_BASELINE = 0.25

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_series(eval_files):
    """
    Returns {model_name: {n: {m: accuracy}}}
    so we can plot one curve per (model, m) with n on the x-axis.
    """
    series = {}
    for path in sorted(eval_files):
        data = load_eval_file(path)
        model = data.get("model_name", os.path.splitext(os.path.basename(path))[0])
        short = model.split("/")[-1] if "/" in model else model

        acc_by_m_n = {}
        for key, d in data.get("per_m_n", {}).items():
            try:
                m_str, n_str = key.split("x")
                m, n = int(m_str), int(n_str)
            except ValueError:
                continue
            if d.get("accuracy") is not None:
                acc_by_m_n[(m, n)] = d["accuracy"]

        if acc_by_m_n:
            series[short] = acc_by_m_n

    return series


# ---------------------------------------------------------------------------
# Plot 1 — Line plot
# ---------------------------------------------------------------------------

def plot_line(series, output_dir, title=""):
    all_n = sorted({n for acc in series.values() for (m, n) in acc})
    all_m = sorted({m for acc in series.values() for (m, n) in acc})

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (model, acc_by_mn) in enumerate(sorted(series.items())):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        for j, m in enumerate(all_m):
            ys = [acc_by_mn.get((m, n), float("nan")) for n in all_n]
            # Only label the first m-curve per model to avoid legend clutter
            label = f"{model} (m={m})" if len(all_m) > 1 else model
            linestyle = ["-", "--", ":", "-."][j % 4]
            ax.plot(range(len(all_n)), ys,
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2, markersize=8)

    ax.set_xticks(range(len(all_n)))
    ax.set_xticklabels([str(n) for n in all_n], fontsize=11)
    ax.set_xlabel("n (number of swaps)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.axhline(DEFAULT_BASELINE, color="gray", linestyle="--", linewidth=1.5,
               label=f"Random baseline ({DEFAULT_BASELINE:.2f})")
    ax.set_title(title or "Accuracy vs n", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(output_dir, f"accuracy_vs_n_line.{ext}")
        plt.savefig(p, dpi=300); print(f"  Saved → {p}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2 — Bar chart
# ---------------------------------------------------------------------------

# def plot_bar(series, output_dir, title=""):
#     all_n = sorted({n for acc in series.values() for (m, n) in acc})
#     all_m = sorted({m for acc in series.values() for (m, n) in acc})
#     model_list = sorted(series.keys())

#     # One group per n value; within each group, bars for each (model, m) combo
#     curve_keys = [(model, m) for model in model_list for m in all_m]
#     n_curves = len(curve_keys)
#     n_groups = len(all_n)

#     x = np.arange(n_groups)
#     width = max(0.08, 0.7 / n_curves)

#     fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.5), 5))

#     for i, (model, m) in enumerate(curve_keys):
#         ys = [series[model].get((m, n), 0.0) for n in all_n]
#         off = (i - n_curves / 2 + 0.5) * width
#         bars = ax.bar(x + off, ys, width,
#                       label=f"{model} (m={m})",
#                       color=COLORS[i % len(COLORS)],
#                       alpha=0.85)
#         for bar, y in zip(bars, ys):
#             if y > 0:
#                 ax.text(bar.get_x() + bar.get_width() / 2, y + 0.01,
#                         f"{y:.2f}", ha="center", va="bottom", fontsize=8,
#                         color=COLORS[i % len(COLORS)])

#     ax.axhline(DEFAULT_BASELINE, color="gray", linestyle="--", linewidth=1.5,
#                label=f"Random ({DEFAULT_BASELINE:.2f})")
#     ax.set_xticks(x)
#     ax.set_xticklabels([str(n) for n in all_n], fontsize=11)
#     ax.set_xlabel("n (number of swaps)", fontsize=12)
#     ax.set_ylabel("Accuracy", fontsize=12)
#     ax.set_ylim(0, 1.12)
#     ax.set_title(title or "Accuracy vs n", fontsize=13, fontweight="bold")
#     ax.legend(fontsize=9)
#     plt.tight_layout()

#     for ext in ("pdf", "png"):
#         p = os.path.join(output_dir, f"accuracy_vs_n_bar.{ext}")
#         plt.savefig(p, dpi=300); print(f"  Saved → {p}")
#     plt.close()


# ---------------------------------------------------------------------------
# Plot 3 — Heatmap (models × difficulties)
# ---------------------------------------------------------------------------

# def plot_heatmap(series, output_dir, title=""):
#     # Rows = (model, m) combos; columns = n values
#     all_n = sorted({n for acc in series.values() for (m, n) in acc})
#     all_m = sorted({m for acc in series.values() for (m, n) in acc})
#     model_list = sorted(series.keys())

#     row_keys = [(model, m) for model in model_list for m in all_m]
#     grid = np.full((len(row_keys), len(all_n)), np.nan)

#     for i, (model, m) in enumerate(row_keys):
#         for j, n in enumerate(all_n):
#             val = series[model].get((m, n))
#             if val is not None:
#                 grid[i, j] = val

#     row_labels = [f"{model} m={m}" for model, m in row_keys]

#     fig, ax = plt.subplots(figsize=(max(6, len(all_n) * 1.2), max(4, len(row_keys) * 0.8)))
#     im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

#     ax.set_xticks(range(len(all_n)))
#     ax.set_xticklabels([str(n) for n in all_n], fontsize=10)
#     ax.set_yticks(range(len(row_keys)))
#     ax.set_yticklabels(row_labels, fontsize=10)
#     ax.set_xlabel("n (number of swaps)", fontsize=12)
#     ax.set_title(title or "Accuracy Heatmap (Models × n)", fontsize=13, fontweight="bold")

#     for i in range(len(row_keys)):
#         for j in range(len(all_n)):
#             val = grid[i, j]
#             if not np.isnan(val):
#                 ax.text(j, i, f"{val:.2f}", ha="center", va="center",
#                         fontsize=9, color="black")

#     plt.colorbar(im, ax=ax, label="Accuracy")
#     plt.tight_layout()

#     for ext in ("pdf", "png"):
#         p = os.path.join(output_dir, f"accuracy_heatmap.{ext}")
#         plt.savefig(p, dpi=300); print(f"  Saved → {p}")
#     plt.close()
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from eval.py output files."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Paths (or globs) to eval JSON files.",
    )
    parser.add_argument(
        "--output-dir", default="figures",
        help="Directory to write figures (default: figures/).",
    )
    parser.add_argument(
        "--title", default="",
        help="Optional figure title suffix.",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Skip the heatmap (useful with many models).",
    )
    args = parser.parse_args()

    # Expand any globs
    eval_files = []
    for pat in args.inputs:
        matches = glob.glob(pat)
        eval_files.extend(matches if matches else [pat])
    eval_files = sorted(set(eval_files))

    if not eval_files:
        print("No eval files found.", file=__import__("sys").stderr)
        raise SystemExit(1)

    print(f"Loading {len(eval_files)} eval file(s) …")
    for f in eval_files:
        print(f"  {f}")

    series = collect_series(eval_files)
    if not series:
        print("No plottable data found in the provided eval files.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nModels: {sorted(series.keys())}")
    print(f"Output : {args.output_dir}\n")

    plot_line(series, args.output_dir, title=args.title)
    # plot_bar(series,  args.output_dir, title=args.title)
    # if not args.no_heatmap:
    #     plot_heatmap(series, args.output_dir, title=args.title)

    print("\nAll figures written.")


if __name__ == "__main__":
    main()
