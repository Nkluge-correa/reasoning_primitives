# Reasoning Primitives in Hybrid and Non-Hybrid LLMs: Do Architectural Differences Yield Advantages in State-Tracking and Recall?

Code, data-generation utilities, and plotting scripts for our working paper on **reasoning primitives**. The project studies two core primitives, **recall** and **state-tracking**, and their composition as **state-based recall**: retrieving information whose address must first be inferred by tracking a changing state.

The current paper framing is close to the *State over Tokens* view of reasoning: reasoning traces are effective because they externalize intermediate computational state into token space, letting the model recondition on partial results across decoding steps. In our case study, reasoning augmentation is the clearest and most consistent driver of performance gains, while any hybrid advantage is conditional and task-dependent rather than general.

## Study at a glance

- **Research question:** when reasoning is decomposed into recall and state-tracking, where do reasoning traces matter most, and when do architectural differences between transformers and hybrids become visible?
- **Model setup:** matched `OLMo-3-7B` and `OLMo-3-Hybrid-7B` families, each with Instruct and Think variants.
- **Difficulty axes:** `m` controls retrieval complexity and `n` controls state-maintenance complexity.
- **Evaluation:** deterministic post-hoc scoring from free-form generation, with both exact accuracy and parsed weighted accuracy (PWA).

## Tasks

| Name             | Description                                                                                    | `m`                           | `n`                       |
|------------------|------------------------------------------------------------------------------------------------|-------------------------------|---------------------------|
| `olmo_original`  | Minimal baseline from Merrill et al. (2026): track pointer swaps, then index into a bit array  | bit-array size                | number of swap lines      |
| `astro`          | Track variable-to-planet mappings through swaps over a real exoplanet table                    | number of table rows          | number of swaps           |
| `collisions`     | Track particle velocities through sequential elastic collisions                                | number of particles           | number of collision steps |
| `dyck`           | Track a bracket stack through a Dyck expression and identify the correct masked closer         | stack depth at query position | sequence length           |
| `dag_arithmetic` | Trace addition/subtraction over a DAG with cross-layer dependencies and report a queried value | variables per layer (width)   | number of layers (depth)  |

Most tasks use **4-option multiple choice (A/B/C/D)**; `dyck` uses **3 options (A/B/C)** because there are only three valid closing brackets.

## Repository structure

```
├── manuscript/                  # Working paper, references, and manuscript assets
├── figures/                     # Publication figures and generated plots
├── src/
│   ├── utils.py                 # Shared helpers (JSON I/O, answer parsing, vLLM loading)
│   ├── templates.py             # Task definitions, prompts, and generators
│   ├── generator.py             # Generate benchmark datasets (no GPU needed)
│   ├── inference.py             # Run a model over a dataset (requires GPU + vLLM)
│   ├── inference.sh             # SLURM job script for inference
│   ├── eval.py                  # Compute accuracy and parsed weighted accuracy
│   ├── paper_plots.py           # Generate manuscript figures from eval outputs
│   └── task_token_counter.py    # Check prompt token lengths across difficulty levels
├── data/                        # Generated datasets (auto-created)
└── results/                     # Inference and evaluation outputs
```

Adding a new task in `templates.py` automatically integrates with generation, inference, evaluation, and plotting.

## Installation

**Requirements**

- Python 3.10+
- Install from the project root using the extras defined in `pyproject.toml`.
- For **generation and evaluation only** (no GPU needed).

To install the required dependencies, run:

```bash
pip install -e .
```

- For **inference** (GPU required):

```bash
pip install -e ".[inference]"
```

- For **plotting**:

```bash
pip install -e ".[plots]"
```

- For **everything in this repo** (inference, plotting, and tests):

```bash
pip install -e ".[full]"
```

## Example workflow

The full experimental loop is simple:

1. Clone the repo and install the extra you need from `pyproject.toml`.
2. If you are running `astro`, make `exoplanets.csv` available in `src/`, set `EXOPLANETS_CSV`, or pass `--csv-path`.
3. Generate a dataset locally with `generator.py`, choosing a task plus the `m` and `n` difficulty values you want.
4. Optionally run `task_token_counter.py` before inference to size `--max-model-len`, especially for one-shot datasets and Think models.
5. Run inference locally or through `inference.sh` on a cluster.
6. Score outputs with `eval.py`.
7. Plot the eval files with `paper_plots.py`.

A compact end-to-end example:

```bash
# Clone and install
git clone https://github.com/ultor1996/reasoning_primitives.git
cd reasoning_primitives
pip install -e ".[full]"

# Generate data
python generator.py --task olmo_original --n-samples 100 --m 4 8 16 32 64

# Optional: check prompt lengths before inference
python task_token_counter.py --task olmo_original

# Run inference
python inference.py \
    --input data/olmo_original_m4_8_16_32_64_n4_8_16_32_64_s100.json \
    --model allenai/OLMo-3-7B-Think \
    --output results/olmo_original_olmo3_think.json

# Evaluate
python eval.py \
    --input results/olmo_original_olmo3_think.json \
    --output results/olmo_original_olmo3_think_eval.json

# Plot
python paper_plots.py \
    --inputs results/olmo_original_olmo3_think_eval.json \
    --output-dir figures/
```

Notes:
- Use `--mode cartesian` in `generator.py` when you want to vary `m` and `n` independently.
- For Think models, `--max-tokens 4000` is usually more appropriate than the instruct-style default.
- For cluster runs, update `HF_CACHE` in `src/inference.sh` and submit the same arguments through `sbatch inference.sh ...`.

## Adding a new task

All tasks are defined in `templates.py`. To add one:

1. Write a generator function `my_task_generator(m: int, n: int, rng: random.Random) -> dict` that returns a dict with keys `prompt`, `correct_option`, `option_A`-`option_D`, and `metadata`.
2. Add a system prompt string and optionally a one-shot example string.
3. Register it in `_REGISTRY` at the bottom of the file, passing both strings.

That's it — `generator.py`, `inference.py`, and `eval.py` will all pick it up automatically. The `--shot one` flag will work for the new task immediately if a one-shot example is provided.

If you are extending the benchmark for the paper, keep the task design principle fixed: the task should isolate recall, state-tracking, or their composition in a way that exposes how performance changes as `m` and `n` scale.


## Citation

```bibtex
@misc{rawat2026reasoningprimitiveshybridnonhybrid,
      title={Reasoning Primitives in Hybrid and Non-Hybrid LLMs: Do Architectural Differences Yield Advantages in State-Tracking and Recall?}, 
      author={Shivam Rawat and Lucie Flek and Florian Mai and Nicholas Kluge Corrêa},
      year={2026},
      eprint={2604.21454},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.21454}, 
}
```