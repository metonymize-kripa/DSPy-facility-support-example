"""
07_evaluation/failure_analysis.py

Cross-approach failure analysis: which examples are hard across all approaches?

For each test example, counts how many approaches got it wrong. Examples wrong
by all (or most) approaches are structurally hard — they represent the irreducible
difficulty of the task or ambiguity in the labels.

Also:
  - Identifies the "easy wins" — examples that simple rule-based classifiers get right
    but LLMs get wrong (over-engineering failures)
  - Identifies where GEPA-optimized prompts diverge from zero-shot LLMs

Run:
    uv run 07_evaluation/failure_analysis.py [--top N]

Prints hard case analysis to stdout and saves to data/results/failure_analysis.json.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import load_examples, split_examples
from shared.metrics import CATEGORIES as ALL_CATEGORIES

# Which result files to compare, and what predict functions to reconstruct
# For failure analysis we need to re-run predictions — but we only have stored
# aggregate metrics, not per-example predictions. We'll re-import the predict
# functions directly.

APPROACH_MODULES = [
    ("Handwritten rules",       "01_handwritten_rules.rules",         "predict"),
    ("FastText",                None,                                  None),  # needs models on disk
    ("Classical ML (LogReg)",   "03_classical_ml.train_eval",          None),  # needs fit
]

# Since re-running all approaches is expensive, this script takes a different approach:
# It loads each per-example result from stored JSON files that include failures,
# and performs the cross-approach analysis from those.

RESULT_FILES = {
    "Handwritten rules":       "01_handwritten_rules.json",
    "FastText":                "02_fasttext.json",
    "Classical ML":            "03_classical_ml_best.json",
    "spaCy entity-augmented":  "04_spacy_entity_augmented.json",
    "spaCy textcat":           "04_spacy_textcat.json",
    "gemma4:26b zero-shot":    "06_llm_parent_base.json",
    "gemma4:e4b zero-shot":    "06_llm_student_base.json",
    "gemma4:e4b + GEPA":       "06_llm_student_compiled.json",
}


def load_failures(path: Path) -> list[dict] | None:
    """Load the failures list from a result JSON."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("failures", None)
    except Exception:
        return None


def analyze_cross_approach_failures(
    failures_by_approach: dict[str, list[dict]],
    test_examples: list[dict],
    top_n: int = 10,
) -> dict:
    """
    For each test example, count how many approaches failed on it.
    Returns sorted list of hardest examples.
    """
    # Failures are stored as list of {message, true_*, predicted_*}
    # We need to identify examples by message text

    msg_to_idx = {ex["message"]: i for i, ex in enumerate(test_examples)}

    # Count failures per example
    failure_count = defaultdict(int)
    failure_details = defaultdict(dict)

    for approach, failures in failures_by_approach.items():
        if failures is None:
            continue
        for f in failures:
            msg = f.get("message", "")
            idx = msg_to_idx.get(msg)
            if idx is None:
                continue
            failure_count[idx] += 1
            failure_details[idx][approach] = {
                "true_urgency":    f.get("true_urgency"),
                "pred_urgency":    f.get("predicted_urgency"),
                "true_sentiment":  f.get("true_sentiment"),
                "pred_sentiment":  f.get("predicted_sentiment"),
                "true_categories": f.get("true_categories"),
                "pred_categories": f.get("predicted_categories"),
            }

    # Sort by failure count descending
    sorted_failures = sorted(failure_count.items(), key=lambda x: x[1], reverse=True)

    hard_cases = []
    for idx, count in sorted_failures[:top_n]:
        ex = test_examples[idx]
        hard_cases.append({
            "idx": idx,
            "failure_count": count,
            "n_approaches": len(failures_by_approach),
            "fraction_failed": round(count / len(failures_by_approach), 3),
            "message_preview": ex["message"][:200],
            "true_urgency": ex["urgency"],
            "true_sentiment": ex["sentiment"],
            "true_categories": ex["categories"],
            "per_approach": failure_details[idx],
        })

    return hard_cases


def analyze_over_engineering(
    failures_by_approach: dict[str, list[dict]],
    test_examples: list[dict],
) -> list[dict]:
    """
    Find examples where simple approaches succeed but LLMs fail.
    The "over-engineering failure" pattern.
    """
    simple_approaches = {"Handwritten rules", "FastText", "Classical ML"}
    llm_approaches = {"gemma4:26b zero-shot", "gemma4:e4b zero-shot", "gemma4:e4b + GEPA"}

    msg_to_idx = {ex["message"]: i for i, ex in enumerate(test_examples)}

    simple_failures = set()
    llm_failures = set()

    for approach, failures in failures_by_approach.items():
        if failures is None:
            continue
        for f in failures:
            idx = msg_to_idx.get(f.get("message", ""))
            if idx is None:
                continue
            if approach in simple_approaches:
                simple_failures.add(idx)
            elif approach in llm_approaches:
                llm_failures.add(idx)

    # Simple succeeds, LLM fails
    simple_wins = llm_failures - simple_failures
    cases = []
    for idx in sorted(simple_wins):
        ex = test_examples[idx]
        cases.append({
            "idx": idx,
            "message_preview": ex["message"][:200],
            "true_urgency":    ex["urgency"],
            "true_sentiment":  ex["sentiment"],
            "true_categories": ex["categories"],
            "note": "Simple approach correct, LLM failed",
        })
    return cases


def analyze_gepa_wins(
    failures_by_approach: dict[str, list[dict]],
    test_examples: list[dict],
) -> dict:
    """Cases where GEPA-compiled model succeeds but zero-shot fails."""
    msg_to_idx = {ex["message"]: i for i, ex in enumerate(test_examples)}

    zeroshot_failures = set()
    gepa_failures = set()

    for approach, failures in failures_by_approach.items():
        if failures is None:
            continue
        if approach == "gemma4:e4b zero-shot":
            for f in failures:
                idx = msg_to_idx.get(f.get("message", ""))
                if idx is not None:
                    zeroshot_failures.add(idx)
        elif approach == "gemma4:e4b + GEPA":
            for f in failures:
                idx = msg_to_idx.get(f.get("message", ""))
                if idx is not None:
                    gepa_failures.add(idx)

    gepa_wins = zeroshot_failures - gepa_failures
    gepa_regressions = gepa_failures - zeroshot_failures

    return {
        "gepa_wins_count": len(gepa_wins),
        "gepa_regression_count": len(gepa_regressions),
        "gepa_wins": [
            {"idx": i, "preview": test_examples[i]["message"][:150]}
            for i in sorted(gepa_wins)[:5]
        ],
        "gepa_regressions": [
            {"idx": i, "preview": test_examples[i]["message"][:150]}
            for i in sorted(gepa_regressions)[:5]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="Show top N hardest examples")
    parser.add_argument("--results-dir", default="data/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    examples = load_examples()
    _, _, test_examples = split_examples(examples)

    print(f"Test set: {len(test_examples)} examples")

    # Load failures from each approach
    failures_by_approach = {}
    for approach, filename in RESULT_FILES.items():
        failures = load_failures(results_dir / filename)
        failures_by_approach[approach] = failures
        status = f"{len(failures)} failures" if failures is not None else "not found"
        print(f"  {approach:<35} {status}")

    available = {k: v for k, v in failures_by_approach.items() if v is not None}
    if not available:
        print("\nNo failure data found. Result JSONs must include a 'failures' list.")
        print("Check that approach scripts use shared.metrics.evaluate() which stores failures.")
        sys.exit(0)

    print(f"\n{len(available)} approaches with failure data available.")

    # Cross-approach hard cases
    print("\n" + "="*70)
    print(f"TOP {args.top} HARDEST EXAMPLES (most approaches failed)")
    print("="*70)

    hard_cases = analyze_cross_approach_failures(available, test_examples, top_n=args.top)

    for i, case in enumerate(hard_cases, 1):
        print(f"\n{i}. Example #{case['idx']} — failed by {case['failure_count']}/{case['n_approaches']} approaches ({case['fraction_failed']:.0%})")
        print(f"   True labels: urgency={case['true_urgency']}, sentiment={case['true_sentiment']}")
        print(f"   Categories: {', '.join(case['true_categories'])}")
        print(f"   Message: {case['message_preview'][:120]}...")

    # Over-engineering failures
    print("\n" + "="*70)
    print("OVER-ENGINEERING FAILURES (rules correct, LLM wrong)")
    print("="*70)

    oe_cases = analyze_over_engineering(available, test_examples)
    print(f"Found {len(oe_cases)} examples where simple approaches beat LLMs.")
    for case in oe_cases[:5]:
        print(f"\n  Example #{case['idx']}: {case['message_preview'][:100]}...")
        print(f"  True: urgency={case['true_urgency']}, sentiment={case['true_sentiment']}")

    # GEPA wins/losses
    print("\n" + "="*70)
    print("GEPA vs. ZERO-SHOT (gemma4:e4b)")
    print("="*70)

    gepa = analyze_gepa_wins(available, test_examples)
    print(f"GEPA wins (0-shot failed, GEPA correct): {gepa['gepa_wins_count']}")
    print(f"GEPA regressions (0-shot correct, GEPA failed): {gepa['gepa_regression_count']}")

    if gepa["gepa_wins"]:
        print("\n  Sample GEPA wins:")
        for w in gepa["gepa_wins"]:
            print(f"    #{w['idx']}: {w['preview'][:80]}...")

    if gepa["gepa_regressions"]:
        print("\n  Sample GEPA regressions:")
        for r in gepa["gepa_regressions"]:
            print(f"    #{r['idx']}: {r['preview'][:80]}...")

    # Save
    output = {
        "n_test": len(test_examples),
        "approaches_analyzed": list(available.keys()),
        "hard_cases": hard_cases,
        "over_engineering_failures": len(oe_cases),
        "over_engineering_sample": oe_cases[:5],
        "gepa_analysis": gepa,
    }
    out_path = results_dir / "failure_analysis.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
