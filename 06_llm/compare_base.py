# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "dspy-ai"]
# ///
"""
06_llm/compare_base.py
-----------------------
Zero-shot comparison: gemma4:26b (parent) vs gemma4:e4b (student).
No GEPA optimisation — establishes the raw gap that GEPA needs to close.

Run:
    uv run --group llm 06_llm/compare_base.py
    uv run --group llm 06_llm/compare_base.py --test-size 20  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy

from shared.dataset import CATEGORIES, load_examples, split_examples
from shared.metrics import evaluate, print_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student-model", default="ollama/gemma4:e4b")
    p.add_argument("--parent-model", default="ollama/gemma4:26b")
    p.add_argument("--api-base", default="http://localhost:11434")
    p.add_argument("--api-key", default="ollama")
    p.add_argument("--student-num-ctx", type=int, default=8192)
    p.add_argument("--parent-num-ctx", type=int, default=32768)
    p.add_argument("--test-size", type=int, default=0,
                   help="Number of test examples to evaluate (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--timeout-s", type=int, default=1200)
    p.add_argument("--think", choices=["off", "on", "unset"], default="off")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DSPy program (same structure as archive/dspy_support_analyzer.py)
# ---------------------------------------------------------------------------

class UrgencySig(dspy.Signature):
    """Read the provided facility-support message and determine the urgency."""
    message: str = dspy.InputField()
    urgency: Literal["low", "medium", "high"] = dspy.OutputField()


class SentimentSig(dspy.Signature):
    """Read the provided facility-support message and determine the sentiment."""
    message: str = dspy.InputField()
    sentiment: Literal["positive", "neutral", "negative"] = dspy.OutputField()


class CategoriesSig(dspy.Signature):
    """Read the provided facility-support message and determine all applicable categories."""
    message: str = dspy.InputField()
    categories: List[Literal[
        "emergency_repair_services", "routine_maintenance_requests",
        "quality_and_safety_concerns", "specialized_cleaning_services",
        "general_inquiries", "sustainability_and_environmental_practices",
        "training_and_support_requests", "cleaning_services_scheduling",
        "customer_feedback_and_complaints", "facility_management_issues",
    ]] = dspy.OutputField()


class FacilitySupportAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.urgency_module = dspy.ChainOfThought(UrgencySig)
        self.sentiment_module = dspy.ChainOfThought(SentimentSig)
        self.categories_module = dspy.ChainOfThought(CategoriesSig)

    def forward(self, message: str) -> dspy.Prediction:
        import re, ast
        u = self.urgency_module(message=message)
        s = self.sentiment_module(message=message)
        c = self.categories_module(message=message)
        return dspy.Prediction(
            urgency=_norm(u.urgency, {"low", "medium", "high"}),
            sentiment=_norm(s.sentiment, {"positive", "neutral", "negative"}),
            categories=_norm_cats(c.categories),
        )


def _norm(value: Any, allowed: set) -> str:
    import re
    text = str(value or "").strip().lower().strip("`'\" ")
    if text in allowed:
        return text
    for label in allowed:
        if re.search(rf"\b{label}\b", text):
            return label
    return text


def _norm_cats(value: Any) -> list[str]:
    import ast, json, re
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            try:
                value = ast.literal_eval(value)
            except Exception:
                value = [x.strip() for x in value.strip("[]").split(",") if x.strip()]
    if isinstance(value, dict):
        items = [k for k, v in value.items() if v]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    cats = set(CATEGORIES)
    return [str(i).strip().lower().strip("`'\" ") for i in items
            if str(i).strip().lower().strip("`'\" ") in cats]


def make_lm(model: str, api_base: str, api_key: str, num_ctx: int, think: str) -> dspy.LM:
    kwargs: dict[str, Any] = {
        "api_base": api_base, "api_key": api_key,
        "cache": False, "num_ctx": num_ctx,
    }
    if think == "off":
        kwargs["think"] = False
    elif think == "on":
        kwargs["think"] = True
    try:
        return dspy.LM(model, **kwargs)
    except TypeError:
        kwargs.pop("think", None)
        return dspy.LM(model, **kwargs)


def run_model(lm, label: str, test_set: list[dict], args) -> dict:
    dspy.settings.configure(lm=lm, num_threads=args.num_threads, timeout_s=args.timeout_s)
    program = FacilitySupportAnalyzer()

    def predict(message: str) -> dict:
        pred = program(message=message)
        return {
            "urgency": pred.urgency,
            "sentiment": pred.sentiment,
            "categories": pred.categories,
        }

    t0 = time.perf_counter()
    results = evaluate(test_set, predict)
    elapsed = time.perf_counter() - t0

    print_results(label, results)
    print(f"  Elapsed: {elapsed:.1f}s  ({elapsed/len(test_set):.1f}s/example)")
    return results


def main():
    args = parse_args()

    examples = load_examples()
    _, _, test = split_examples(examples, seed=args.seed)
    if args.test_size > 0:
        test = test[:args.test_size]
    print(f"Evaluating on {len(test)} test examples")

    results_path = Path(__file__).parent.parent / "data" / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_id, label, num_ctx, result_key in [
        (args.student_model, f"gemma4:e4b (student, zero-shot)", args.student_num_ctx, "06_llm_student_base"),
        (args.parent_model, f"gemma4:26b (parent, zero-shot)", args.parent_num_ctx, "06_llm_parent_base"),
    ]:
        print(f"\n── {label} ──")
        lm = make_lm(model_id, args.api_base, args.api_key, num_ctx, args.think)
        results = run_model(lm, label, test, args)
        all_results[result_key] = results

        out = {
            "approach": "llm_gemma4",
            "variant": result_key,
            "model": model_id,
            "n_test": results["n"],
            "aggregate": round(results["aggregate"], 4),
            "urgency": round(results["urgency"], 4),
            "sentiment": round(results["sentiment"], 4),
            "categories": round(results["categories"], 4),
            "cost_per_query_usd": 0.0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "training_examples_required": 0,
            "notes": f"zero-shot; think={args.think}; num_ctx={num_ctx}",
        }
        with open(results_path / f"{result_key}.json", "w") as f:
            json.dump(out, f, indent=2)

    # Print gap
    if "06_llm_student_base" in all_results and "06_llm_parent_base" in all_results:
        sb = all_results["06_llm_student_base"]
        pb = all_results["06_llm_parent_base"]
        print(f"\n── Gap (parent − student) ──")
        print(f"  Aggregate:  {(pb['aggregate'] - sb['aggregate'])*100:+.1f}pp")
        print(f"  Urgency:    {(pb['urgency']   - sb['urgency']  )*100:+.1f}pp")
        print(f"  Sentiment:  {(pb['sentiment'] - sb['sentiment'])*100:+.1f}pp")
        print(f"  Categories: {(pb['categories']- sb['categories'])*100:+.1f}pp")
        print(f"\nThis gap is what GEPA optimisation aims to close.")
        print(f"Run gepa_optimise.py next.")


if __name__ == "__main__":
    main()
