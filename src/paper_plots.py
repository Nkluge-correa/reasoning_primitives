# """
# paper_plots.py — generate paper-quality figures from eval.py output files.

# Usage
# -----
# # Single eval file
# python paper_plots.py --inputs results/*_eval.json --output-dir figures/

# # Or compare across tasks
# python paper_plots.py \\
#     --inputs results/collisions_*_eval.json \\
#     --group-by task \\
#     --output-dir figures/

# Figures produced
# ----------------
# 1. accuracy_vs_difficulty_line.pdf  — line plot, one curve per model
# 2. accuracy_vs_difficulty_bar.pdf   — grouped bar chart
# 3. accuracy_heatmap.pdf             — heatmap (models × difficulties)

# All figures are also saved as .png (300 dpi) for quick preview.
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
# # Style constants
# # ---------------------------------------------------------------------------

# COLORS  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#FF5722", "#795548", "#00ACC1"]
# MARKERS = ["o",        "s",       "^",       "D",       "v",       "P",       "X",       "*"]

# RANDOM_BASELINE = {4: 0.25}          # 4-option MC -> 0.25; 2-option -> 0.50
# DEFAULT_BASELINE = 0.25
# DYCK_BASELINE = 0.33 

# plt.rcParams.update({
#     "font.family": "sans-serif",
#     "axes.spines.top": False,
#     "axes.spines.right": False,
#     "axes.grid": True,
#     "grid.alpha": 0.3,
# })

# # ---------------------------------------------------------------------------
# # Data loading
# # ---------------------------------------------------------------------------

# def load_eval_file(path: str) -> dict:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def collect_series(eval_files, max_m=None):
#     # First pass: accumulate values per (model, pair) across seeds
#     raw = {}        # {short_model: {pair: [val1, val2, ...]}}
#     tasks_found = set()

#     for path in sorted(eval_files):
#         data = load_eval_file(path)
#         model = data.get("model_name", os.path.splitext(os.path.basename(path))[0])
#         short = model.split("/")[-1] if "/" in model else model
#         # shot is stored per-sample, not at top level — read from first sample
#         samples = data.get("scored_samples") or data.get("samples") or []
#         if samples:
#             shot = samples[0].get("shot", "zero")
#         else:
#             # fallback: infer from filename
#             fname = os.path.basename(path)
#             shot = "one" if "oneshot" in fname else "zero"
#         short = f"{short} ({shot})"
#         task  = data.get("task", "unknown")
#         tasks_found.add(task)

#         if short not in raw:
#             raw[short] = {
#                 "accuracy":   {},
#                 "pwa":        {},
#                 "parse_rate": {},
#             }

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

#     # Second pass: compute mean and std across seeds
#     series = {}
#     for short, metrics in raw.items():
#         acc_mean, acc_std     = {}, {}
#         pwa_mean, pwa_std     = {}, {}
#         rate_mean, rate_std   = {}, {}

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

#     return series, tasks_found# ← return tasks too

# # ---------------------------------------------------------------------------
# # Plot 1 — Line plot
# # ---------------------------------------------------------------------------

# def plot_line(series, output_dir, title="", task=None):
#     all_pairs = sorted({pair for s in series.values() for pair in s["accuracy"]})
#     x_labels  = [f"({m},{n})" for m, n in all_pairs]

#     fig, ax = plt.subplots(figsize=(max(9, len(all_pairs) * 0.8), 5))

#     for i, (model, s) in enumerate(sorted(series.items())):
#         color  = COLORS[i % len(COLORS)]
#         marker = MARKERS[i % len(MARKERS)]
#         xs     = list(range(len(all_pairs)))
#         ys     = [s["accuracy"].get(pair, float("nan")) for pair in all_pairs]
#         ys_std = [s["accuracy_std"].get(pair, 0.0) for pair in all_pairs]

#         ax.plot(xs, ys, label=model, color=color,
#                 marker=marker, linestyle="-", linewidth=2, markersize=8)

#         ys_arr  = np.array(ys,     dtype=float)
#         std_arr = np.array(ys_std, dtype=float)
#         ax.fill_between(xs,
#                         np.clip(ys_arr - std_arr, 0, 1),
#                         np.clip(ys_arr + std_arr, 0, 1),
#                         alpha=0.15, color=color)

#         for x, y in zip(xs, ys):
#             if not np.isnan(y):
#                 ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 9),
#                             textcoords="offset points",
#                             ha="center", fontsize=7, color=color)

#     ax.set_xticks(range(len(all_pairs)))
#     ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")
#     ax.set_xlabel("(m, n)", fontsize=12)
#     ax.set_ylabel("Accuracy", fontsize=12)
#     ax.set_ylim(0, 1.10)
#     # ax.axhline(DEFAULT_BASELINE, color="gray", linestyle="--", linewidth=1.5,
#     #            label=f"Random baseline ({DEFAULT_BASELINE:.2f})")
#     baseline = DYCK_BASELINE if task and "dyck" in task else DEFAULT_BASELINE
#     ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.5,
#            label=f"Random baseline ({baseline:.2f})")
#     ax.set_title(title or "Accuracy vs Difficulty", fontsize=13, fontweight="bold")
#     ax.legend(fontsize=9, loc="upper right")
#     plt.tight_layout()
#     model_names = "_".join(sorted(series.keys()))
#     task_str = f"{task}_" if task else ""
#     for ext in ("pdf", "png"):
#         p = os.path.join(output_dir, f"accuracy_line_{task_str}{model_names}.{ext}")
#         plt.savefig(p, dpi=300); print(f"  Saved → {p}")
#     plt.close()

# def plot_pwa_line(series, output_dir, title="", task=None):
#     all_pairs = sorted({pair for s in series.values() for pair in s["pwa"]})
#     x_labels  = [f"({m},{n})" for m, n in all_pairs]

#     fig, ax = plt.subplots(figsize=(max(9, len(all_pairs) * 0.8), 5))

#     # Replace the plotting loop inside plot_pwa_line with:
#     for i, (model, s) in enumerate(sorted(series.items())):
#         color  = COLORS[i % len(COLORS)]
#         marker = MARKERS[i % len(MARKERS)]
#         xs     = list(range(len(all_pairs)))
#         ys     = [s["pwa"].get(pair, float("nan")) for pair in all_pairs]
#         ys_std = [s["pwa_std"].get(pair, 0.0) for pair in all_pairs]

#         ax.plot(xs, ys, label=model, color=color,
#                 marker=marker, linestyle="-", linewidth=2, markersize=8)

#         ys_arr  = np.array(ys,     dtype=float)
#         std_arr = np.array(ys_std, dtype=float)
#         ax.fill_between(xs,
#                         np.clip(ys_arr - std_arr, 0, 1),
#                         np.clip(ys_arr + std_arr, 0, 1),
#                         alpha=0.15, color=color)

#         for x, y in zip(xs, ys):
#             if not np.isnan(y):
#                 ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 9),
#                             textcoords="offset points",
#                             ha="center", fontsize=7, color=color)

#     ax.set_xticks(range(len(all_pairs)))
#     ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")
#     ax.set_xlabel("(m, n)", fontsize=12)
#     ax.set_ylabel("Parsed Weighted Accuracy", fontsize=12)
#     ax.set_ylim(0, 1.10)
#     # ax.axhline(DEFAULT_BASELINE, color="gray", linestyle="--", linewidth=1.5,
#     #            label=f"Random baseline ({DEFAULT_BASELINE:.2f})")
#     baseline = DYCK_BASELINE if task and "dyck" in task else DEFAULT_BASELINE
#     ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.5,
#            label=f"Random baseline ({baseline:.2f})")
#     ax.set_title(title or "Parsed Weighted Accuracy vs Difficulty",
#                  fontsize=13, fontweight="bold")
#     ax.legend(fontsize=9, loc="upper right")
#     plt.tight_layout()
#     model_names = "_".join(sorted(series.keys()))
#     task_str = f"{task}_" if task else ""
#     for ext in ("pdf", "png"):
#         p = os.path.join(output_dir, f"pwa_line_{task_str}{model_names}.{ext}")
#         plt.savefig(p, dpi=300); print(f"  Saved → {p}")
#     plt.close()


# def plot_parse_rate(series, output_dir, title="", task=None):
#     all_pairs = sorted({pair for s in series.values() for pair in s["parse_rate"]})
#     x_labels  = [f"({m},{n})" for m, n in all_pairs]

#     fig, ax = plt.subplots(figsize=(max(9, len(all_pairs) * 0.8), 5))

#     # Replace the plotting loop inside plot_parse_rate with:
#     for i, (model, s) in enumerate(sorted(series.items())):
#         color  = COLORS[i % len(COLORS)]
#         marker = MARKERS[i % len(MARKERS)]
#         xs     = list(range(len(all_pairs)))
#         ys     = [s["parse_rate"].get(pair, float("nan")) for pair in all_pairs]
#         ys_std = [s["parse_rate_std"].get(pair, 0.0) for pair in all_pairs]

#         ax.plot(xs, ys, label=model, color=color,
#                 marker=marker, linestyle="-", linewidth=2, markersize=8)

#         ys_arr  = np.array(ys,     dtype=float)
#         std_arr = np.array(ys_std, dtype=float)
#         ax.fill_between(xs,
#                         np.clip(ys_arr - std_arr, 0, 1),
#                         np.clip(ys_arr + std_arr, 0, 1),
#                         alpha=0.15, color=color)

#         for x, y in zip(xs, ys):
#             if not np.isnan(y):
#                 ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 9),
#                             textcoords="offset points",
#                             ha="center", fontsize=7, color=color)

#     ax.set_xticks(range(len(all_pairs)))
#     ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")
#     ax.set_xlabel("(m, n)", fontsize=12)
#     ax.set_ylabel("Parse Rate (n_scored / n_total)", fontsize=12)
#     ax.set_ylim(0, 1.10)
#     ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5,
#                label="Perfect parse rate (1.0)")
#     ax.set_title(title or "Parse Rate vs Difficulty", fontsize=13, fontweight="bold")
#     ax.legend(fontsize=9, loc="lower left")
#     plt.tight_layout()
#     model_names = "_".join(sorted(series.keys()))
#     task_str = f"{task}_" if task else ""
#     for ext in ("pdf", "png"):
#         p = os.path.join(output_dir, f"parse_rate_line_{task_str}{model_names}.{ext}")
#         plt.savefig(p, dpi=300); print(f"  Saved → {p}")
#     plt.close()

# # ---------------------------------------------------------------------------
# # Plot 2 — Bar chart
# # ---------------------------------------------------------------------------

# # def plot_bar(series, output_dir, title=""):
# #     all_n = sorted({n for acc in series.values() for (m, n) in acc})
# #     all_m = sorted({m for acc in series.values() for (m, n) in acc})
# #     model_list = sorted(series.keys())

# #     # One group per n value; within each group, bars for each (model, m) combo
# #     curve_keys = [(model, m) for model in model_list for m in all_m]
# #     n_curves = len(curve_keys)
# #     n_groups = len(all_n)

# #     x = np.arange(n_groups)
# #     width = max(0.08, 0.7 / n_curves)

# #     fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.5), 5))

# #     for i, (model, m) in enumerate(curve_keys):
# #         ys = [series[model].get((m, n), 0.0) for n in all_n]
# #         off = (i - n_curves / 2 + 0.5) * width
# #         bars = ax.bar(x + off, ys, width,
# #                       label=f"{model} (m={m})",
# #                       color=COLORS[i % len(COLORS)],
# #                       alpha=0.85)
# #         for bar, y in zip(bars, ys):
# #             if y > 0:
# #                 ax.text(bar.get_x() + bar.get_width() / 2, y + 0.01,
# #                         f"{y:.2f}", ha="center", va="bottom", fontsize=8,
# #                         color=COLORS[i % len(COLORS)])

# #     ax.axhline(DEFAULT_BASELINE, color="gray", linestyle="--", linewidth=1.5,
# #                label=f"Random ({DEFAULT_BASELINE:.2f})")
# #     ax.set_xticks(x)
# #     ax.set_xticklabels([str(n) for n in all_n], fontsize=11)
# #     ax.set_xlabel("n (number of swaps)", fontsize=12)
# #     ax.set_ylabel("Accuracy", fontsize=12)
# #     ax.set_ylim(0, 1.12)
# #     ax.set_title(title or "Accuracy vs n", fontsize=13, fontweight="bold")
# #     ax.legend(fontsize=9)
# #     plt.tight_layout()

# #     for ext in ("pdf", "png"):
# #         p = os.path.join(output_dir, f"accuracy_vs_n_bar.{ext}")
# #         plt.savefig(p, dpi=300); print(f"  Saved → {p}")
# #     plt.close()


# # ---------------------------------------------------------------------------
# # Plot 3 — Heatmap (models × difficulties)
# # ---------------------------------------------------------------------------

# # def plot_heatmap(series, output_dir, title=""):
# #     # Rows = (model, m) combos; columns = n values
# #     all_n = sorted({n for acc in series.values() for (m, n) in acc})
# #     all_m = sorted({m for acc in series.values() for (m, n) in acc})
# #     model_list = sorted(series.keys())

# #     row_keys = [(model, m) for model in model_list for m in all_m]
# #     grid = np.full((len(row_keys), len(all_n)), np.nan)

# #     for i, (model, m) in enumerate(row_keys):
# #         for j, n in enumerate(all_n):
# #             val = series[model].get((m, n))
# #             if val is not None:
# #                 grid[i, j] = val

# #     row_labels = [f"{model} m={m}" for model, m in row_keys]

# #     fig, ax = plt.subplots(figsize=(max(6, len(all_n) * 1.2), max(4, len(row_keys) * 0.8)))
# #     im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

# #     ax.set_xticks(range(len(all_n)))
# #     ax.set_xticklabels([str(n) for n in all_n], fontsize=10)
# #     ax.set_yticks(range(len(row_keys)))
# #     ax.set_yticklabels(row_labels, fontsize=10)
# #     ax.set_xlabel("n (number of swaps)", fontsize=12)
# #     ax.set_title(title or "Accuracy Heatmap (Models × n)", fontsize=13, fontweight="bold")

# #     for i in range(len(row_keys)):
# #         for j in range(len(all_n)):
# #             val = grid[i, j]
# #             if not np.isnan(val):
# #                 ax.text(j, i, f"{val:.2f}", ha="center", va="center",
# #                         fontsize=9, color="black")

# #     plt.colorbar(im, ax=ax, label="Accuracy")
# #     plt.tight_layout()

# #     for ext in ("pdf", "png"):
# #         p = os.path.join(output_dir, f"accuracy_heatmap.{ext}")
# #         plt.savefig(p, dpi=300); print(f"  Saved → {p}")
# #     plt.close()
# # ---------------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate paper figures from eval.py output files."
#     )
#     parser.add_argument(
#         "--inputs", nargs="+", required=True,
#         help="Paths (or globs) to eval JSON files.",
#     )
#     parser.add_argument(
#         "--output-dir", default="figures",
#         help="Directory to write figures (default: figures/).",
#     )
#     parser.add_argument(
#         "--title", default="",
#         help="Optional figure title suffix.",
#     )
#     parser.add_argument(
#         "--no-heatmap", action="store_true",
#         help="Skip the heatmap (useful with many models).",
#     )
#     parser.add_argument(
#     "--max-m", type=int, default=None,
#     help="Cap plots at this m value (e.g. 2048 to exclude larger difficulties).",
#     )
#     parser.add_argument(
#         "--task", default=None,
#         help="Task name to include in output filenames (e.g. olmo_original, dyck).",
#     )
#     args = parser.parse_args()

#     # Expand any globs
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
#         print("No plottable data found in the provided eval files.")
#         return

#     # Use CLI --task if provided, otherwise auto-detect from JSON
#     task_str = args.task or ("_".join(sorted(tasks_found)) if tasks_found else "unknown")

#     os.makedirs(args.output_dir, exist_ok=True)
#     print(f"\nModels: {sorted(series.keys())}")
#     print(f"Task   : {task_str}")
#     print(f"Output : {args.output_dir}\n")

#     plot_line(series, args.output_dir, title=args.title, task=task_str)
#     plot_pwa_line(series, args.output_dir, title=args.title, task=task_str)
#     plot_parse_rate(series, args.output_dir, title=args.title, task=task_str)
#     # plot_bar(series,  args.output_dir, title=args.title)
#     # if not args.no_heatmap:
#     #     plot_heatmap(series, args.output_dir, title=args.title)

#     print("\nAll figures written.")


# if __name__ == "__main__":
#     main()

"""
paper_plots.py — generate EMNLP-style paper figures from eval.py output files.

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
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# EMNLP / ACL style constants
# ---------------------------------------------------------------------------

COLORS = [
    "#2166AC",   # blue      — instruct
    "#D6604D",   # red       — think
    "#4DAC26",   # green     — hybrid instruct
    "#E08B00",   # amber     — hybrid think
    "#762A83",   # purple
    "#1B7837",   # dark green
    "#BF812D",   # brown
    "#35978F",   # teal
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

RANDOM_BASELINE  = {4: 0.25}
DEFAULT_BASELINE = 0.25
DYCK_BASELINE    = 0.33

plt.rcParams.update({
    # --- typography (ACL / EMNLP papers use Times) ---
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          13,
    "axes.labelsize":     14,
    "axes.titlesize":     14,
    "legend.fontsize":    8.5,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    # --- clean frame ---
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    # --- grid ---
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.4,
    "grid.color":         "#cccccc",
    # --- output ---
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# Shared legend kwargs — compact, no long line handles
LEGEND_KW = dict(
    framealpha=0.9,
    edgecolor="#cccccc",
    ncol=1,
    handlelength=0.8,
    handletextpad=0.3,
    borderpad=0.4,
    labelspacing=0.25,
    markerscale=0.65,
)

# ---------------------------------------------------------------------------
# Data loading  (unchanged from original)
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
# Shared helper
# ---------------------------------------------------------------------------

def _annotate_values(ax, xs, ys, color):
    """Print value labels above each data point."""
    for x, y in zip(xs, ys):
        if not np.isnan(y):
            ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 7),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8.5, color=color, fontweight="bold")


def _save(fig, output_dir, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(p)
        print(f"  Saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — Accuracy line
# ---------------------------------------------------------------------------

def plot_line(series, output_dir, title="", task=None):
    all_pairs = sorted({pair for s in series.values() for pair in s["accuracy"]})
    x_labels  = [f"({m},{n})" for m, n in all_pairs]
    xs        = list(range(len(all_pairs)))

    fig, ax = plt.subplots(figsize=(max(9, len(all_pairs) * 0.8), 4.8))

    baseline = DYCK_BASELINE if task and "dyck" in task else DEFAULT_BASELINE
    ax.axhline(baseline, color="#888888", linestyle="--", linewidth=1.2,
               label=f"Random baseline ({baseline:.2f})", zorder=1)

    for i, (model, s) in enumerate(sorted(series.items())):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ys     = [s["accuracy"].get(pair, float("nan")) for pair in all_pairs]
        ys_std = [s["accuracy_std"].get(pair, 0.0)      for pair in all_pairs]

        ax.plot(xs, ys, label=model, color=color,
                marker=marker, linewidth=1.8, markersize=7, zorder=3)

        ys_arr  = np.array(ys,     dtype=float)
        std_arr = np.array(ys_std, dtype=float)
        if std_arr.any():
            ax.fill_between(xs,
                            np.clip(ys_arr - std_arr, 0, 1),
                            np.clip(ys_arr + std_arr, 0, 1),
                            alpha=0.15, color=color)

        _annotate_values(ax, xs, ys, color)

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel("(m, n)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="lower left", **LEGEND_KW)
    # title intentionally omitted for paper style

    plt.tight_layout()
    task_str    = f"{task}_" if task else ""
    model_names = "_".join(sorted(series.keys()))
    _save(fig, output_dir, f"accuracy_line_{task_str}{model_names}")


# ---------------------------------------------------------------------------
# Plot 2 — Parsed Weighted Accuracy line
# ---------------------------------------------------------------------------

def plot_pwa_line(series, output_dir, title="", task=None):
    all_pairs = sorted({pair for s in series.values() for pair in s["pwa"]})
    x_labels  = [f"({m},{n})" for m, n in all_pairs]
    xs        = list(range(len(all_pairs)))

    fig, ax = plt.subplots(figsize=(max(9, len(all_pairs) * 0.8), 4.8))

    baseline = DYCK_BASELINE if task and "dyck" in task else DEFAULT_BASELINE
    ax.axhline(baseline, color="#888888", linestyle="--", linewidth=1.2,
               label=f"Random baseline ({baseline:.2f})", zorder=1)

    for i, (model, s) in enumerate(sorted(series.items())):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ys     = [s["pwa"].get(pair, float("nan")) for pair in all_pairs]
        ys_std = [s["pwa_std"].get(pair, 0.0)      for pair in all_pairs]

        ax.plot(xs, ys, label=model, color=color,
                marker=marker, linewidth=1.8, markersize=7, zorder=3)

        ys_arr  = np.array(ys,     dtype=float)
        std_arr = np.array(ys_std, dtype=float)
        if std_arr.any():
            ax.fill_between(xs,
                            np.clip(ys_arr - std_arr, 0, 1),
                            np.clip(ys_arr + std_arr, 0, 1),
                            alpha=0.15, color=color)

        _annotate_values(ax, xs, ys, color)

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel("(m, n)")
    ax.set_ylabel("Parsed Weighted Accuracy")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="lower left", **LEGEND_KW)

    plt.tight_layout()
    task_str    = f"{task}_" if task else ""
    model_names = "_".join(sorted(series.keys()))
    _save(fig, output_dir, f"pwa_line_{task_str}{model_names}")


# ---------------------------------------------------------------------------
# Plot 3 — Parse rate line
# ---------------------------------------------------------------------------

def plot_parse_rate(series, output_dir, title="", task=None):
    all_pairs = sorted({pair for s in series.values() for pair in s["parse_rate"]})
    x_labels  = [f"({m},{n})" for m, n in all_pairs]
    xs        = list(range(len(all_pairs)))

    fig, ax = plt.subplots(figsize=(max(9, len(all_pairs) * 0.8), 4.8))

    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.2,
               label="Perfect parse rate (1.0)", zorder=1)

    for i, (model, s) in enumerate(sorted(series.items())):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ys     = [s["parse_rate"].get(pair, float("nan")) for pair in all_pairs]
        ys_std = [s["parse_rate_std"].get(pair, 0.0)      for pair in all_pairs]

        ax.plot(xs, ys, label=model, color=color,
                marker=marker, linewidth=1.8, markersize=7, zorder=3)

        ys_arr  = np.array(ys,     dtype=float)
        std_arr = np.array(ys_std, dtype=float)
        if std_arr.any():
            ax.fill_between(xs,
                            np.clip(ys_arr - std_arr, 0, 1),
                            np.clip(ys_arr + std_arr, 0, 1),
                            alpha=0.15, color=color)

        _annotate_values(ax, xs, ys, color)

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel("(m, n)")
    ax.set_ylabel("Parse Rate (n_scored / n_total)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="lower left", **LEGEND_KW)

    plt.tight_layout()
    task_str    = f"{task}_" if task else ""
    model_names = "_".join(sorted(series.keys()))
    _save(fig, output_dir, f"parse_rate_line_{task_str}{model_names}")


# ---------------------------------------------------------------------------
# CLI  (unchanged from original)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate EMNLP-style paper figures from eval.py output files."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths (or globs) to eval JSON files.")
    parser.add_argument("--output-dir", default="figures",
                        help="Directory to write figures (default: figures/).")
    parser.add_argument("--title", default="",
                        help="Optional figure title suffix (unused in EMNLP style).")
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

    print(f"Loading {len(eval_files)} eval file(s) …")
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