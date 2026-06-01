"""
06_llm/thinking_ablation.py

Ablation: gemma4:e4b with think=True vs think=False.

gemma4 models support an extended thinking mode where the model reasons through
the problem before producing the final answer. This script tests whether forcing
explicit chain-of-thought improves classification accuracy on this narrow,
structured task — or whether it hurts by spending tokens on reasoning that isn't
needed for a well-defined taxonomy.

Run:
    uv run --group llm 06_llm/thinking_ablation.py

Results saved to:
    data/results/06_llm_thinking_ablation.json
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["dspy-ai>=2.5", "python-dotenv>=1.0"]
# ///

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from dotenv import load_dotenv

from shared.dataset import load_examples, split_examples
from shared.metrics import evaluate, print_results

load_dotenv()

# ---------------------------------------------------------------------------
# DSPy signatures (same as compare_base.py)
# ---------------------------------------------------------------------------

CATEGORIES = [
    "routine_maintenance_requests",
    "customer_feedback_and_complaints",
    "training_and_support_requests",
    "quality_and_safety_concerns",
    "sustainability_and_environmental_practices",
    "cleaning_services_scheduling",
    "specialized_cleaning_services",
    "emergency_repair_services",
    "facility_management_issues",
    "general_inquiries",
]


class ClassifyUrgency(dspy.Signature):
    """Classify the urgency of a facility support message as low, medium, or high."""
    message: str = dspy.InputField(desc="The facility support message text")
    urgency: str = dspy.OutputField(desc="Urgency level: low, medium, or high")


class ClassifySentiment(dspy.Signature):
    """Classify the sentiment of a facility support message as positive, neutral, or negative."""
    message: str = dspy.InputField(desc="The facility support message text")
    sentiment: str = dspy.OutputField(desc="Sentiment: positive, neutral, or negative")


class ClassifyCategories(dspy.Signature):
    """Identify all applicable service categories for a facility support message.

    Categories: routine_maintenance_requests, customer_feedback_and_complaints,
    training_and_support_requests, quality_and_safety_concerns,
    sustainability_and_environmental_practices, cleaning_services_scheduling,
    specialized_cleaning_services, emergency_repair_services,
    facility_management_issues, general_inquiries.

    Return a comma-separated list of all matching categories.
    """
    message: str = dspy.InputField(desc="The facility support message text")
    categories: str = dspy.OutputField(desc="Comma-separated list of applicable categories")


class FacilitySupportClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.urgency_module = dspy.Predict(ClassifyUrgency)
        self.sentiment_module = dspy.Predict(ClassifySentiment)
        self.categories_module = dspy.Predict(ClassifyCategories)

    def forward(self, message: str):
        urgency_result = self.urgency_module(message=message)
        sentiment_result = self.sentiment_module(message=message)
        categories_result = self.categories_module(message=message)
        return dspy.Prediction(
            urgency=urgency_result.urgency.strip().lower(),
            sentiment=sentiment_result.sentiment.strip().lower(),
            categories=categories_result.categories,
        )


def make_lm(think: bool):
    """Create gemma4:e4b LM with specified thinking mode."""
    kwargs = dict(
        model="ollama/gemma4:e4b",
        api_base="http://localhost:11434",
        temperature=0.0,
        max_tokens=1024,
    )
    if not think:
        kwargs["extra_body"] = {"think": False}
    return dspy.LM(**kwargs)


def parse_categories(raw: str) -> list[str]:
    parts = [p.strip().lower().replace(" ", "_") for p in raw.split(",")]
    return [p for p in parts if p in CATEGORIES]


def run_condition(label: str, think: bool, examples: list[dict]) -> dict:
    lm = make_lm(think=think)
    dspy.configure(lm=lm)

    classifier = FacilitySupportClassifier()

    def predict_fn(message: str) -> dict:
        pred = classifier(message=message)
        return {
            "urgency": pred.urgency,
            "sentiment": pred.sentiment,
            "categories": parse_categories(pred.categories),
        }

    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"  think={think}")
    print(f"  model=ollama/gemma4:e4b")
    print(f"{'='*60}")

    start = time.time()
    results = evaluate(examples, predict_fn)
    elapsed = time.time() - start

    print_results(label, results)
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/len(examples):.1f}s/example)")

    return {
        "label": label,
        "model": "ollama/gemma4:e4b",
        "think": think,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_example": round(elapsed / len(examples), 2),
        **results,
    }


def main():
    examples = load_examples()
    _, _, test_examples = split_examples(examples)

    print(f"Test set: {len(test_examples)} examples")
    print("Ablation: think=False vs think=True on gemma4:e4b")
    print("Hypothesis: thinking mode may not help on a well-defined taxonomy task")
    print("           and may hurt by consuming tokens on unnecessary reasoning")

    results = {}

    # Condition 1: think=False (baseline, recommended for structured tasks)
    no_think = run_condition("gemma4:e4b (think=False)", think=False, examples=test_examples)
    results["no_think"] = no_think

    # Condition 2: think=True (extended chain-of-thought)
    with_think = run_condition("gemma4:e4b (think=True)", think=True, examples=test_examples)
    results["with_think"] = with_think

    # Delta analysis
    print("\n" + "="*60)
    print("DELTA: think=True minus think=False")
    print("="*60)
    for metric in ["urgency", "sentiment", "categories", "aggregate"]:
        delta = with_think[metric] - no_think[metric]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:12s}: {sign}{delta:.4f}")

    speed_ratio = with_think["seconds_per_example"] / no_think["seconds_per_example"]
    print(f"\n  Speed ratio (think=True / think=False): {speed_ratio:.1f}x slower")

    # Paper interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    agg_delta = with_think["aggregate"] - no_think["aggregate"]
    if abs(agg_delta) < 0.02:
        print("  Result: No meaningful accuracy difference.")
        print("  Thinking mode provides negligible benefit on this structured taxonomy task.")
        print("  Recommendation: Use think=False for cost/speed efficiency.")
    elif agg_delta > 0.02:
        print(f"  Result: Thinking mode improves aggregate by {agg_delta:.3f}.")
        print("  Unexpected: explicit CoT helps even on a narrow classification task.")
    else:
        print(f"  Result: Thinking mode HURTS aggregate by {abs(agg_delta):.3f}.")
        print("  Thinking tokens crowd out classification signal on a narrow task.")

    # Save
    output = {
        "experiment": "thinking_ablation",
        "model": "ollama/gemma4:e4b",
        "n_test": len(test_examples),
        "conditions": results,
    }
    out_path = Path("data/results/06_llm_thinking_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
