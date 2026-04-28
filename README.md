# State-Based Recall Evaluation Suite

A pipeline for generating, running, and evaluating LLM benchmarks focused on **state tracking** and **state-based recall** — the reasoning primitives studied in [OLMo Hybrid (Merrill et al., 2026)](https://arxiv.org/abs/2604.03444).

---

## Tasks

| Name | Description | Difficulty axis |
|---|---|---|
| `collisions` | Track particle velocities through elastic collisions | `m` particles = `n` collision steps |
| `astro` | Track variable→planet mappings through swap operations over an exoplanet table | `m` table rows = `n` swaps |
| `olmo_original` | State-based recall from the OLMo Hybrid paper: track 5 pointer variables through swaps, then index into a bit array | `n` swaps = `m` bit-array size |

All tasks output **4-option multiple choice (A/B/C/D)**.

---

## Repository structure

```
├── utils.py          # Shared helpers (JSON I/O, answer parsing, vLLM loading)
├── templates.py      # Task definitions — all prompts and generators live here
├── generator.py      # Generate evaluation datasets (no GPU needed)
├── inference.py      # Run a model over a dataset (requires GPU + vLLM)
├── inference.sh      # SLURM job script for HPC clusters
├── eval.py           # Compute accuracy from inference output
├── paper_plots.py    # Generate publication figures
└── data/             # Default output directory for generated datasets
```

---

## Installation

### Requirements

- Python 3.10+
- For **generation and evaluation only** (no GPU needed):

```bash
pip install json-repair
```

- For **inference** (GPU required):

```bash
pip install vllm transformers accelerate torch json-repair
```

### Clone the repo

```bash
git clone <your-repo-url>
cd <repo-name>
```

### Exoplanet CSV (for `astro` task only)

The `astro` task requires an exoplanet CSV file. Set the path in one of two ways:

```bash
# Option 1 — environment variable (persists for the session)
export EXOPLANETS_CSV=/path/to/exoplanets.csv

# Option 2 — pass it directly at generation time
python generator.py --task astro --csv-path /path/to/exoplanets.csv ...
```

The CSV must contain these columns:
`Planet`, `Host Star`, `Orbital Period (days)`, `Planet Radius (Earth radii)`, `Planet Mass (Earth masses)`, `Equilibrium Temp (K)`, `Semi-Major Axis (AU)`, `Eccentricity`, `Stellar Temp (K)`, `Stellar Radius (Solar radii)`, `Stellar Mass (Solar masses)`

---

## Step 1 — Generate a dataset

Run locally (no GPU needed). Output goes to `data/` by default.

```bash
# Collision task — 100 samples at each of 4 difficulty levels
python generator.py \
    --task collisions \
    --n-samples 100 \
    --difficulties 4 8 16 32

# OLMo original state-based recall task
python generator.py \
    --task olmo_original \
    --n-samples 100 \
    --difficulties 4 8 16 32 64

# Astro task with explicit CSV path
python generator.py \
    --task astro \
    --n-samples 100 \
    --difficulties 4 8 16 \
    --csv-path /path/to/exoplanets.csv
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--task` | required | `collisions`, `astro`, or `olmo_original` |
| `--n-samples` | `100` | Samples per difficulty level |
| `--difficulties` | `4 8 16` | Space-separated list of difficulty values |
| `--output-dir` | `data/` inside repo | Directory to save the JSON file |
| `--output` | auto-named | Override the output filename |
| `--csv-path` | env var / compiled default | Path to exoplanet CSV (`astro` only) |
| `--seed` | `42` | Random seed for reproducibility |

Output is a single JSON file, e.g. `data/collisions_diff4_8_16_n100.json`.

---

## Step 2 — Run inference (HPC / GPU)

### On a local GPU

```bash
python inference.py \
    --input  data/collisions_diff4_8_16_n100.json \
    --model  allenai/Olmo-3-7B-Think \
    --output results/collisions_olmo3think.json
```

### On a SLURM cluster (Marvin)

```bash
sbatch inference.sh \
    --input  /path/to/data/collisions_diff4_8_16_n100.json \
    --model  allenai/Olmo-3-7B-Think \
    --output /path/to/results/collisions_olmo3think.json
```

Before submitting, check these two variables at the top of `inference.sh` match your cluster setup:

```bash
CONDA_ENV="your_env_name"
HF_CACHE="/path/to/.cache/huggingface"
```

**All inference options:**

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to generator output JSON |
| `--model` | required | HuggingFace model name or local path |
| `--output` | auto-named | Output JSON path |
| `--batch-size` | `8` | vLLM generation batch size |
| `--max-model-len` | `16000` | Max context + generation length |
| `--tensor-parallel` | `1` | Number of GPUs for tensor parallelism |

The output file is the input JSON with three extra fields added to every sample: `raw_output`, `completion` (thinking trace stripped), and `reasoning` (content of `<think>…</think>` if present).

---

## Step 3 — Evaluate

```bash
python eval.py \
    --input  results/collisions_olmo3think.json \
    --output scores/collisions_olmo3think_eval.json
```

Prints a summary to stdout and writes a JSON file with overall accuracy and per-difficulty breakdown.

Add `--no-samples` to omit per-sample detail and keep the output file small.

---

## Step 4 — Plot results

```bash
python paper_plots.py \
    --inputs scores/*_eval.json \
    --output-dir figures/
```

Produces three figures (line plot, bar chart, heatmap) in both `.pdf` and `.png`.

---

## End-to-end example

```bash
# 1. Generate
python generator.py --task olmo_original --n-samples 100 --difficulties 4 8 16 32 64

# 2. Copy to cluster and run inference
scp data/olmo_original_diff4_8_16_32_64_n100.json marvin:/path/to/data/
sbatch inference.sh \
    --input  /path/to/data/olmo_original_diff4_8_16_32_64_n100.json \
    --model  allenai/Olmo-Hybrid-Think-SFT-7B \
    --output /path/to/results/sbr_hybrid_think.json

# 3. Copy results back and evaluate
scp marvin:/path/to/results/sbr_hybrid_think.json results/
python eval.py --input results/sbr_hybrid_think.json

# 4. Plot
python paper_plots.py --inputs results/*_eval.json --output-dir figures/
```

---

## Adding a new task

All tasks are defined in `templates.py`. To add one:

1. Write a generator function `my_task_generator(difficulty: int, rng: random.Random) -> dict` that returns a dict with keys `prompt`, `correct_option`, `option_A`–`option_D`, and `metadata`.
2. Add a system prompt string.
3. Register it in `_REGISTRY` at the bottom of the file.

That's it — `generator.py`, `inference.py`, and `eval.py` will all pick it up automatically.

---

## Citation

If you use the `olmo_original` task, please cite:

```bibtex
@article{merrill2026olmohybrid,
  title   = {OLMo Hybrid: From Theory to Practice and Back},
  author  = {Merrill, William and Li, Yanhong and Romero, Tyler and others},
  journal = {arXiv preprint arXiv:2604.03444},
  year    = {2026}
}
```
