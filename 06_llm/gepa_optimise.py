# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "dspy-ai", "gepa"]
# ///
"""
06_llm/gepa_optimise.py
------------------------
Full GEPA optimisation loop: gemma4:e4b student, gemma4:26b reflection.
Produces three-way comparison: parent/base vs student/base vs student/compiled.

Ported and restructured from archive/dspy_support_analyzer.py.

Quick smoke test:
    uv run --group llm 06_llm/gepa_optimise.py \
      --train-size 4 --val-size 2 --gepa-max-metric-calls 6

Paper-quality run:
    uv run --group llm 06_llm/gepa_optimise.py \
      --train-size 20 --val-size 10 --gepa-max-metric-calls 30 --comparison-size 68
"""

from __future__ import annotations

import argparse
import inspect
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
    p.add_argument("--reflection-model", default="ollama/gemma4:26b")
    p.add_argument("--api-base", default="http://localhost:11434")
    p.add_argument("--api-key", default="ollama")
    p.add_argument("--student-num-ctx", type=int, default=8192)
    p.add_argument("--reflection-num-ctx", type=int, default=32768)
    p.add_argument("--student-thinking", choices=["off", "on", "unset"], default="off")
    p.add_argument("--reflection-thinking", choices=["off", "on", "unset"], default="unset")
    p.add_argument("--train-size", type=int, default=10)
    p.add_argument("--val-size", type=int, default=5)
    p.add_argument("--gepa-max-metric-calls", type=int, default=15)
    p.add_argument("--comparison-size", type=int, default=10)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--timeout-s", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# DSPy program
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
    import ast, json as _json
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = _json.loads(value)
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


# ---------------------------------------------------------------------------
# GEPA metric with per-predictor feedback
# (ported from archive/gepa_gemma_minimal.py — minimal, no hand-crafted policy)
# ---------------------------------------------------------------------------

GEPA_SCOPE = (
    "When revising a predictor, optimize only that predictor's declared output field "
    "using the provided gold-vs-prediction feedback; preserve the original input/output "
    "schema and do not add unrelated tasks, fields, examples, or formatting requirements. "
    "importantly: less is more, avoid prompt bloat at all costs. "
)


def _gepa_feedback(score: float, feedback: str) -> dspy.Prediction:
    return dspy.Prediction(score=float(score), feedback=GEPA_SCOPE + feedback)


def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    gold = json.loads(example["answer"])

    pred_urgency = _norm(getattr(pred, "urgency", ""), {"low", "medium", "high"})
    pred_sentiment = _norm(getattr(pred, "sentiment", ""), {"positive", "neutral", "negative"})

    gold_cat_set = {k for k, v in gold["categories"].items() if v}
    pred_cat_set = set(_norm_cats(getattr(pred, "categories", [])))

    u_score = 1.0 if gold["urgency"] == pred_urgency else 0.0
    s_score = 1.0 if gold["sentiment"] == pred_sentiment else 0.0
    correct = sum(
        1 for cat in CATEGORIES
        if (cat in gold_cat_set) == (cat in pred_cat_set)
    )
    c_score = correct / len(CATEGORIES)
    total = (u_score + s_score + c_score) / 3.0

    if pred_name is None:
        return total

    if "urgency_module" in pred_name:
        fb = f"predicted={pred_urgency}; gold={gold['urgency']}"
        return _gepa_feedback(u_score, fb)
    if "sentiment_module" in pred_name:
        fb = f"predicted={pred_sentiment}; gold={gold['sentiment']}"
        return _gepa_feedback(s_score, fb)
    if "categories_module" in pred_name:
        fp = sorted(pred_cat_set - gold_cat_set)
        fn = sorted(gold_cat_set - pred_cat_set)
        fb = f"false_positives={fp}; false_negatives={fn}"
        return _gepa_feedback(c_score, fb)

    return _gepa_feedback(total, f"urgency={pred_urgency}/{gold['urgency']} "
                                  f"sentiment={pred_sentiment}/{gold['sentiment']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lm(model: str, api_base, api_key, num_ctx, thinking) -> dspy.LM:
    kwargs: dict[str, Any] = {
        "api_base": api_base, "api_key": api_key,
        "cache": False, "num_ctx": num_ctx,
    }
    if thinking == "off":
        kwargs["think"] = False
    elif thinking == "on":
        kwargs["think"] = True
    try:
        return dspy.LM(model, **kwargs)
    except TypeError:
        kwargs.pop("think", None)
        return dspy.LM(model, **kwargs)


def examples_to_dspy(examples: list[dict]) -> list[dspy.Example]:
    out = []
    for ex in examples:
        answer = json.dumps({
            "urgency": ex["urgency"],
            "sentiment": ex["sentiment"],
            "categories": ex["categories"],
        })
        out.append(
            dspy.Example(message=ex["message"], answer=answer).with_inputs("message")
        )
    return out


def score_program(lm, program, test_set: list[dict], args) -> dict:
    dspy.settings.configure(lm=lm, num_threads=args.num_threads, timeout_s=args.timeout_s)
    def predict(message):
        pred = program(message=message)
        return {"urgency": pred.urgency, "sentiment": pred.sentiment,
                "categories": pred.categories}
    return evaluate(test_set, predict)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Student:    {args.student_model}")
    print(f"Reflection: {args.reflection_model}")
    print(f"Train/val:  {args.train_size}/{args.val_size}")
    print(f"GEPA calls: {args.gepa_max_metric_calls}")

    examples = load_examples()
    train, val, test = split_examples(examples, seed=args.seed)
    trainset = train[:args.train_size]
    valset = val[:args.val_size]
    comparison_set = test[:args.comparison_size]

    print(f"Train: {len(trainset)}  Val: {len(valset)}  Comparison: {len(comparison_set)}")

    student_lm = make_lm(args.student_model, args.api_base, args.api_key,
                          args.student_num_ctx, args.student_thinking)
    reflection_lm = make_lm(args.reflection_model, args.api_base, args.api_key,
                              args.reflection_num_ctx, args.reflection_thinking)

    dspy.settings.configure(lm=student_lm, num_threads=args.num_threads,
                             timeout_s=args.timeout_s)

    # Convert to DSPy examples for GEPA
    dspy_train = examples_to_dspy(trainset)
    dspy_val = examples_to_dspy(valset)

    # GEPA setup
    try:
        GEPA = dspy.GEPA
    except AttributeError:
        from dspy.teleprompt import GEPA

    gepa_kwargs = {
        "metric": metric_with_feedback,
        "reflection_lm": reflection_lm,
        "max_metric_calls": args.gepa_max_metric_calls,
        "num_threads": args.num_threads,
        "track_stats": True,
        "use_merge": False,
    }
    # Filter to accepted kwargs
    sig = inspect.signature(GEPA)
    gepa_kwargs = {k: v for k, v in gepa_kwargs.items() if k in sig.parameters
                   or any(p.kind == inspect.Parameter.VAR_KEYWORD
                          for p in sig.parameters.values())}

    optimizer = GEPA(**gepa_kwargs)

    program = FacilitySupportAnalyzer()

    print("\nRunning GEPA optimisation...")
    t0 = time.perf_counter()
    compiled = optimizer.compile(student=program, trainset=dspy_train, valset=dspy_val)
    elapsed = time.perf_counter() - t0
    print(f"Optimisation complete in {elapsed:.1f}s")

    # Three-way comparison
    results_path = Path(__file__).parent.parent / "data" / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, lm, prog, result_key in [
        ("parent/base (gemma4:26b zero-shot)", reflection_lm, FacilitySupportAnalyzer(), "06_llm_parent_base"),
        ("student/base (gemma4:e4b zero-shot)", student_lm, FacilitySupportAnalyzer(), "06_llm_student_base"),
        ("student/compiled (gemma4:e4b + GEPA)", student_lm, compiled, "06_llm_student_compiled"),
    ]:
        print(f"\n── {label} ──")
        r = score_program(lm, prog, comparison_set, args)
        print_results(label, r)
        rows.append((label, r))

        out = {
            "approach": "llm_gemma4",
            "variant": result_key,
            "model": args.student_model if "student" in label else args.reflection_model,
            "n_test": r["n"],
            "aggregate": round(r["aggregate"], 4),
            "urgency": round(r["urgency"], 4),
            "sentiment": round(r["sentiment"], 4),
            "categories": round(r["categories"], 4),
            "cost_per_query_usd": 0.0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "training_examples_required": len(trainset) if "compiled" in label else 0,
            "notes": label,
        }
        with open(results_path / f"{result_key}.json", "w") as f:
            json.dump(out, f, indent=2)

    # Delta summary
    pb = rows[0][1]; sb = rows[1][1]; sc = rows[2][1]
    print("\n── Deltas ──")
    print(f"  student/compiled − student/base:  {(sc['aggregate']-sb['aggregate'])*100:+.1f}pp (GEPA gain)")
    print(f"  student/compiled − parent/base:   {(sc['aggregate']-pb['aggregate'])*100:+.1f}pp (gap to parent)")
    print(f"  student/base − parent/base:       {(sb['aggregate']-pb['aggregate'])*100:+.1f}pp (raw model gap)")

    # Print evolved prompts
    print("\n── Evolved prompts (copy into FINDINGS.md) ──")
    try:
        for name, predictor in compiled.named_predictors():
            print(f"\n[{name}]")
            if hasattr(predictor, "signature") and hasattr(predictor.signature, "instructions"):
                print(predictor.signature.instructions)
    except Exception as e:
        print(f"Could not extract prompts: {e}")


if __name__ == "__main__":
    main()
