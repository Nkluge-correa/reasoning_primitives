"""
eval.py — compute match-rate accuracy from inference.py output.

Usage
-----
python eval.py \\
    --input  results/collisions_olmo3think.json \\
    --output scores/collisions_olmo3think_eval.json

The output is a JSON file with:
  model_name        : str
  task              : str
  n_total           : int
  n_scored          : int   (samples with a parseable answer AND a ground truth)
  n_correct         : int
  n_parse_failed    : int   (raw_output could not be parsed to A/B/C/D)
  overall_accuracy  : float | null
  per_difficulty    : {difficulty_str: {n_scored, n_correct, accuracy, n_parse_failed}}
"""

import argparse
import json
import os
import sys

from utils import extract_json, load_json, normalize_answer, save_json


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_sample(sample: dict) -> dict:
    """
    Parse the model output in *sample* and compare to the ground truth.

    Returns a dict with keys:
      model_choice  : str | None   normalised predicted label
      correct_option: str | None   ground truth
      is_correct    : bool | None  None when scoring is not possible
      parse_failed  : bool
    """
    raw = sample.get("completion") or sample.get("raw_output") or ""
    correct_option = sample.get("correct_option")

    parsed, _, _ = extract_json(raw)

    parse_failed = parsed is None
    if parse_failed:
        return {
            "model_choice": None,
            "correct_option": correct_option,
            "is_correct": None,
            "parse_failed": True,
        }

    raw_ans = (
        parsed.get("answer")
        or parsed.get("Answer")
        or parsed.get("ANSWER")
        or ""
    )
    model_choice = normalize_answer(raw_ans)

    is_correct = None
    if model_choice is not None and correct_option is not None:
        is_correct = model_choice == correct_option

    return {
        "model_choice": model_choice,
        "correct_option": correct_option,
        "is_correct": is_correct,
        "parse_failed": model_choice is None,
    }


def compute_accuracy(samples: list[dict]) -> dict:
    """
    Score all samples and aggregate into per-difficulty and overall statistics.
    """
    model_name = samples[0].get("model_name", "unknown") if samples else "unknown"
    task       = samples[0].get("task", "unknown") if samples else "unknown"

    n_total       = len(samples)
    n_scored      = 0
    n_correct     = 0
    n_parse_failed = 0

    per_difficulty: dict[str, dict] = {}

    scored_samples = []

    for sample in samples:
        result = score_sample(sample)
        # Attach scores back to sample for the detailed output
        scored = dict(sample)
        scored.update(result)
        scored_samples.append(scored)

        diff_key = str(sample.get("difficulty", "unknown"))
        if diff_key not in per_difficulty:
            per_difficulty[diff_key] = {
                "n_scored": 0, "n_correct": 0,
                "n_parse_failed": 0, "n_total": 0,
            }
        per_difficulty[diff_key]["n_total"] += 1

        if result["parse_failed"]:
            n_parse_failed += 1
            per_difficulty[diff_key]["n_parse_failed"] += 1
            continue

        if result["is_correct"] is not None:
            n_scored += 1
            per_difficulty[diff_key]["n_scored"] += 1
            if result["is_correct"]:
                n_correct += 1
                per_difficulty[diff_key]["n_correct"] += 1

    # Compute per-difficulty accuracy
    for key, d in per_difficulty.items():
        d["accuracy"] = (
            round(d["n_correct"] / d["n_scored"], 4)
            if d["n_scored"] > 0 else None
        )

    overall_acc = round(n_correct / n_scored, 4) if n_scored > 0 else None

    return {
        "model_name": model_name,
        "task": task,
        "n_total": n_total,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_parse_failed": n_parse_failed,
        "overall_accuracy": overall_acc,
        "per_difficulty": per_difficulty,
        "scored_samples": scored_samples,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy from inference.py output."
    )
    parser.add_argument("--input",  required=True, help="Inference JSON (from inference.py).")
    parser.add_argument("--output", default=None,  help="Output eval JSON path.")
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Omit per-sample detail from the output (smaller file).",
    )
    args = parser.parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{stem}_eval.json"

    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")

    samples = load_json(args.input)
    print(f"Loaded {len(samples)} samples.")

    results = compute_accuracy(samples)

    if args.no_samples:
        del results["scored_samples"]

    save_json(results, args.output)

    # Summary to stdout
    print()
    print(f"Model            : {results['model_name']}")
    print(f"Task             : {results['task']}")
    print(f"Total samples    : {results['n_total']}")
    print(f"Scored           : {results['n_scored']}")
    print(f"Parse failures   : {results['n_parse_failed']}")
    print(f"Overall accuracy : {results['overall_accuracy']}")
    print()
    print("Per difficulty:")
    for diff, d in sorted(results["per_difficulty"].items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        print(
            f"  diff={diff:>4}  scored={d['n_scored']:>4}  "
            f"correct={d['n_correct']:>4}  "
            f"acc={d['accuracy'] if d['accuracy'] is not None else 'N/A'}"
        )

    print(f"\nEval results written to {args.output}")


if __name__ == "__main__":
    main()
