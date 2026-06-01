"""
shared/metrics.py
-----------------
Canonical metric functions used by every approach.
All scripts must import from here so results are directly comparable.

The three sub-task scores are kept separate so per-task breakdowns
can be reported alongside the aggregate.

Scoring:
    urgency:    exact match → 0.0 or 1.0
    sentiment:  exact match → 0.0 or 1.0
    categories: 10-way binary accuracy (one score per label, averaged)
                — same formulation as the Meta/DSPy tutorial baseline

The 10-way binary accuracy is preferred over Jaccard for the paper because:
  - It is grounded in the ground truth label structure (10 fixed slots)
  - It penalises false positives and false negatives symmetrically
  - It is directly comparable to the published DSPy tutorial baseline (75.4%)

Aggregate score = mean(urgency, sentiment, categories)  ∈ [0, 1]
"""

from __future__ import annotations

from typing import Any

from shared.dataset import CATEGORIES


# ---------------------------------------------------------------------------
# Per-task scorers
# ---------------------------------------------------------------------------

def score_urgency(gold: str, pred: str) -> float:
    return 1.0 if gold == pred else 0.0


def score_sentiment(gold: str, pred: str) -> float:
    return 1.0 if gold == pred else 0.0


def score_categories(gold: dict[str, bool], pred: list[str] | dict[str, bool]) -> float:
    """
    10-way binary accuracy.
    pred may be a list of active category names or a dict[str, bool].
    """
    pred_set = _to_set(pred)
    correct = sum(
        1 for cat in CATEGORIES
        if bool(gold.get(cat, False)) == (cat in pred_set)
    )
    return correct / len(CATEGORIES)


def score_example(gold: dict, pred: dict) -> dict[str, float]:
    """
    Score one prediction against one gold example.

    gold: dict with keys urgency, sentiment, categories (from dataset.py)
    pred: dict with keys urgency, sentiment, categories

    Returns:
        {
            "urgency":    float,
            "sentiment":  float,
            "categories": float,
            "aggregate":  float,
        }
    """
    u = score_urgency(gold["urgency"], pred.get("urgency", ""))
    s = score_sentiment(gold["sentiment"], pred.get("sentiment", ""))
    c = score_categories(gold["categories"], pred.get("categories", []))
    return {
        "urgency": u,
        "sentiment": s,
        "categories": c,
        "aggregate": (u + s + c) / 3.0,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate(
    examples: list[dict],
    predict_fn,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Run predict_fn over examples and return mean scores.

    predict_fn(message: str) -> dict with keys urgency, sentiment, categories

    Returns:
        {
            "urgency":    float,   # mean exact-match accuracy
            "sentiment":  float,
            "categories": float,   # mean 10-way binary accuracy
            "aggregate":  float,   # mean of the three above
            "n":          int,
            "failures":   int,
        }
    """
    totals = {"urgency": 0.0, "sentiment": 0.0, "categories": 0.0, "aggregate": 0.0}
    failures = 0

    for i, ex in enumerate(examples):
        try:
            pred = predict_fn(ex["message"])
            scores = score_example(ex, pred)
        except Exception as exc:
            failures += 1
            scores = {"urgency": 0.0, "sentiment": 0.0, "categories": 0.0, "aggregate": 0.0}
            if failures <= 3:
                print(f"WARNING: predict_fn failed on example {i}: {exc}")

        for k in totals:
            totals[k] += scores[k]

        if verbose:
            print(
                f"[{i:03d}] agg={scores['aggregate']:.3f} "
                f"u={scores['urgency']:.0f} s={scores['sentiment']:.0f} "
                f"c={scores['categories']:.3f}"
            )

    n = len(examples)
    return {k: (v / n if n else 0.0) for k, v in totals.items()} | {"n": n, "failures": failures}


def print_results(label: str, results: dict[str, float]) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Aggregate:  {results['aggregate']:.1%}")
    print(f"  Urgency:    {results['urgency']:.1%}")
    print(f"  Sentiment:  {results['sentiment']:.1%}")
    print(f"  Categories: {results['categories']:.1%}")
    print(f"  N={results['n']}  failures={results['failures']}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _to_set(pred: Any) -> set[str]:
    if isinstance(pred, dict):
        return {k for k, v in pred.items() if v}
    if isinstance(pred, (list, tuple, set)):
        return set(pred)
    return set()
