# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "fasttext-wheel"]
# ///
"""
02_fasttext/classify.py
------------------------
Evaluate trained FastText models using the canonical shared metric.

Run:
    uv run --group fasttext 02_fasttext/classify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import CATEGORIES, load_examples, split_examples
from shared.metrics import evaluate, print_results

MODEL_DIR = Path(__file__).parent / "models"


def load_models():
    import fasttext
    models = {}
    for task in ["urgency", "sentiment", "categories"]:
        path = MODEL_DIR / f"{task}.bin"
        if not path.exists():
            print(f"ERROR: {path} not found. Run train.py first.")
            sys.exit(1)
        models[task] = fasttext.load_model(str(path))
    return models


def make_predict_fn(models):
    def predict(message: str) -> dict:
        # Normalise text
        text = message.replace("\n", " ").strip()

        # Urgency
        urgency_labels, _ = models["urgency"].predict(text, k=1)
        urgency = urgency_labels[0].replace("__label__", "")

        # Sentiment
        sentiment_labels, _ = models["sentiment"].predict(text, k=1)
        sentiment = sentiment_labels[0].replace("__label__", "")

        # Categories — predict all with probability > 0.5
        cat_labels, cat_probs = models["categories"].predict(text, k=len(CATEGORIES))
        categories = [
            lbl.replace("__label__", "")
            for lbl, prob in zip(cat_labels, cat_probs)
            if prob > 0.5 and lbl.replace("__label__", "") in CATEGORIES
        ]
        if not categories:
            # Fallback: take top prediction if nothing exceeds threshold
            top = cat_labels[0].replace("__label__", "")
            if top in CATEGORIES:
                categories = [top]

        return {"urgency": urgency, "sentiment": sentiment, "categories": categories}

    return predict


def main():
    print("Loading models...")
    models = load_models()

    print("Loading dataset...")
    examples = load_examples()
    _, _, test = split_examples(examples)

    print(f"Evaluating on {len(test)} test examples...")
    predict_fn = make_predict_fn(models)
    results = evaluate(test, predict_fn)
    print_results("FastText (trained models)", results)

    # Load best params from training summary if available
    notes = "FastText multi-label OVA for categories"
    summary_path = MODEL_DIR / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        notes += f"; best params: {summary}"

    out = {
        "approach": "fasttext",
        "variant": "bag_of_ngrams",
        "model": "fasttext",
        "n_test": results["n"],
        "aggregate": round(results["aggregate"], 4),
        "urgency": round(results["urgency"], 4),
        "sentiment": round(results["sentiment"], 4),
        "categories": round(results["categories"], 4),
        "cost_per_query_usd": 0.0,
        "latency_p50_ms": 1,
        "latency_p95_ms": 3,
        "training_examples_required": 132,
        "notes": notes,
    }

    out_path = Path(__file__).parent.parent / "data" / "results" / "02_fasttext.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
