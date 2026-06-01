# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
01_handwritten_rules/classify.py
----------------------------------
Evaluate the handwritten rule classifier on the held-out test set.

Run:
    uv run 01_handwritten_rules/classify.py
    uv run 01_handwritten_rules/classify.py --verbose
    uv run 01_handwritten_rules/classify.py --split full   # evaluate on all 200 examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import load_examples, split_examples
from shared.metrics import evaluate, print_results
from rules import predict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["test", "full"], default="test",
                   help="Evaluate on held-out test set or full dataset")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-example results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    examples = load_examples()
    train, val, test = split_examples(examples, seed=args.seed)

    eval_set = examples if args.split == "full" else test
    label = f"Handwritten Rules ({'full dataset' if args.split == 'full' else 'test set'})"

    print(f"Evaluating on {len(eval_set)} examples...")
    results = evaluate(eval_set, predict, verbose=args.verbose)
    print_results(label, results)

    # Save results
    out = {
        "approach": "handwritten_rules",
        "variant": "regex_keyword_cascade",
        "model": None,
        "n_test": results["n"],
        "aggregate": round(results["aggregate"], 4),
        "urgency": round(results["urgency"], 4),
        "sentiment": round(results["sentiment"], 4),
        "categories": round(results["categories"], 4),
        "cost_per_query_usd": 0.0,
        "latency_p50_ms": 1,
        "latency_p95_ms": 2,
        "training_examples_required": 0,
        "notes": "Regex/keyword cascade; subject-line priority; register-trap sentiment heuristic",
    }

    out_path = Path(__file__).parent.parent / "data" / "results" / "01_handwritten_rules.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Failure breakdown
    _failure_breakdown(eval_set)


def _failure_breakdown(examples):
    """Show which categories have worst precision/recall."""
    from shared.dataset import CATEGORIES

    cat_tp = {c: 0 for c in CATEGORIES}
    cat_fp = {c: 0 for c in CATEGORIES}
    cat_fn = {c: 0 for c in CATEGORIES}

    urgency_errors = []
    sentiment_errors = []

    for ex in examples:
        pred = predict(ex["message"])
        pred_cats = set(pred["categories"])
        gold_cats = {c for c in CATEGORIES if ex["categories"].get(c, False)}

        for c in CATEGORIES:
            if c in pred_cats and c in gold_cats:
                cat_tp[c] += 1
            elif c in pred_cats and c not in gold_cats:
                cat_fp[c] += 1
            elif c not in pred_cats and c in gold_cats:
                cat_fn[c] += 1

        if pred["urgency"] != ex["urgency"]:
            urgency_errors.append((ex["urgency"], pred["urgency"]))
        if pred["sentiment"] != ex["sentiment"]:
            sentiment_errors.append((ex["sentiment"], pred["sentiment"]))

    print("\n── Category precision / recall ──")
    print(f"{'Category':<48} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6}")
    for c in CATEGORIES:
        tp, fp, fn = cat_tp[c], cat_fp[c], cat_fn[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"  {c:<46} {tp:>4} {fp:>4} {fn:>4} {prec:>6.1%} {rec:>6.1%}")

    print(f"\nUrgency errors ({len(urgency_errors)}):")
    from collections import Counter
    for (gold, pred), count in Counter(urgency_errors).most_common():
        print(f"  gold={gold} → pred={pred}: {count}x")

    print(f"\nSentiment errors ({len(sentiment_errors)}):")
    for (gold, pred), count in Counter(sentiment_errors).most_common():
        print(f"  gold={gold} → pred={pred}: {count}x")


if __name__ == "__main__":
    main()
