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


def collect_series(eval_files: list[str]) -> dict[str, dict[int, float]]:
    """
    Returns {model_name: {difficulty: accuracy}}.
    Model name is taken from the JSON; if missing, the filename stem is used.
    """
    series = {}
    for path in sorted(eval_files):
        data = load_eval_file(path)
        model = data.get("model_name", os.path.splitext(os.path.basename(path))[0])
        # Shorten for display
        short = model.split("/")[-1] if "/" in model else model

        acc_by_diff = {}
        for diff_key, d in data.get("per_difficulty", {}).items():
            try:
                diff = int(diff_key)
            except ValueError:
                continue
            if d.get("accuracy") is not None:
                acc_by_diff[diff] = d["accuracy"]

        if acc_by_diff:
            series[short] = acc_by_diff

    return series


# ---------------------------------------------------------------------------
# Plot 1 — Line plot
# ---------------------------------------------------------------------------

def plot_line(series: dict[str, dict[int, float]], output_dir: str, title: str = ""):
    all_diffs = sorted({d for diffs in series.values() for d in diffs})
    x_pos     = list(range(len(all_diffs)))
    x_labels  = [str(d) for d in all_diffs]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (model, acc_by_diff) in enumerate(sorted(series.items())):
        ys = [acc_by_diff.get(d, float("nan")) for d in all_diffs]
        ax.plot(
            x_pos, ys,
            label=model,
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linewidth=2, markersize=8,
        )
        for x, y in zip(x_pos, ys):
            if not np.isnan(y):
                ax.annotate(
                    f"{y:.2f}", xy=(x, y), xytext=(0, 9),
                    textcoords="offset points",
                    ha="center", fontsize=8,
                    color=COLORS[i % len(COLORS)],
                )

    ax.axhline(DEFAULT_BASELINE, color="gray", linestyle="--", linewidth=1.5,
               label=f"Random baseline ({DEFAULT_BASELINE:.2f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Difficulty", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.set_title(title or "Accuracy vs Difficulty", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(output_dir, f"accuracy_vs_difficulty_line.{ext}")
        plt.savefig(p, dpi=300)
        print(f"  Saved → {p}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2 — Bar chart
# ---------------------------------------------------------------------------

def plot_bar(series: dict[str, dict[int, float]], output_dir: str, title: str = ""):
    all_diffs = sorted({d for diffs in series.values() for d in diffs})
    model_list = sorted(series.keys())
    n_models = len(model_list)
    n_diffs  = len(all_diffs)

    x     = np.arange(n_diffs)
    width = max(0.08, 0.7 / n_models)

    fig, ax = plt.subplots(figsize=(max(8, n_diffs * 1.5), 5))

    for i, model in enumerate(model_list):
        acc_by_diff = series[model]
        ys  = [acc_by_diff.get(d, 0.0) for d in all_diffs]
        off = (i - n_models / 2 + 0.5) * width

        bars = ax.bar(
            x + off, ys, width,
            label=model,
            color=COLORS[i % len(COLORS)],
            alpha=0.85,
        )
        for bar, y in zip(bars, ys):
            if y > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y + 0.01,
                    f"{y:.2f}", ha="center", va="bottom", fontsize=8,
                    color=COLORS[i % len(COLORS)],
                )

    ax.axhline(DEFAULT_BASELINE, color="gray", linestyle="--", linewidth=1.5,
               label=f"Random ({DEFAULT_BASELINE:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in all_diffs], fontsize=11)
    ax.set_xlabel("Difficulty", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title(title or "Accuracy vs Difficulty", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(output_dir, f"accuracy_vs_difficulty_bar.{ext}")
        plt.savefig(p, dpi=300)
        print(f"  Saved → {p}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3 — Heatmap (models × difficulties)
# ---------------------------------------------------------------------------

def plot_heatmap(series: dict[str, dict[int, float]], output_dir: str, title: str = ""):
    all_diffs  = sorted({d for diffs in series.values() for d in diffs})
    model_list = sorted(series.keys())

    grid = np.full((len(model_list), len(all_diffs)), np.nan)
    for i, model in enumerate(model_list):
        for j, diff in enumerate(all_diffs):
            val = series[model].get(diff)
            if val is not None:
                grid[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, len(all_diffs) * 1.2), max(4, len(model_list) * 0.8)))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(all_diffs)))
    ax.set_xticklabels([str(d) for d in all_diffs], fontsize=10)
    ax.set_yticks(range(len(model_list)))
    ax.set_yticklabels(model_list, fontsize=10)
    ax.set_xlabel("Difficulty", fontsize=12)
    ax.set_title(title or "Accuracy Heatmap (Models × Difficulty)", fontsize=13, fontweight="bold")

    for i in range(len(model_list)):
        for j in range(len(all_diffs)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black")

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(output_dir, f"accuracy_heatmap.{ext}")
        plt.savefig(p, dpi=300)
        print(f"  Saved → {p}")
    plt.close()


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
    plot_bar(series,  args.output_dir, title=args.title)
    if not args.no_heatmap:
        plot_heatmap(series, args.output_dir, title=args.title)

    print("\nAll figures written.")


if __name__ == "__main__":
    main()
