"""
04_spacy/train_textcat.py

Standard spaCy textcat pipeline: train a multi-label text categorizer using
spaCy's built-in textcat_multilabel component.

This is the "proper" spaCy approach — learned end-to-end — as opposed to the
entity-augmented neuro-symbolic hybrid in entity_augmented.py. Comparing the
two isolates the contribution of the explicit entity features.

Run:
    uv run --group spacy 04_spacy/train_textcat.py

Results saved to:
    data/results/04_spacy_textcat.json
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["spacy>=3.7", "scikit-learn>=1.4"]
# ///

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

from shared.dataset import load_examples, split_examples
from shared.metrics import evaluate, print_results

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

URGENCY_LABELS = ["low", "medium", "high"]
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

TRAIN_EPOCHS = 20
BATCH_SIZE_START = 4.0
BATCH_SIZE_END = 32.0
DROP = 0.2


def make_cats(labels: list[str], all_labels: list[str]) -> dict:
    """Convert list of active labels to spaCy cats dict {label: 0/1}."""
    return {l: 1.0 if l in labels else 0.0 for l in all_labels}


def build_and_train(train_docs, dev_docs, all_labels: list[str], task: str):
    """Build a blank spaCy model with textcat_multilabel and train it."""
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat_multilabel")
    for label in all_labels:
        textcat.add_label(label)

    optimizer = nlp.begin_training()
    train_examples = [
        Example.from_dict(nlp.make_doc(text), {"cats": cats})
        for text, cats in train_docs
    ]
    dev_examples = [
        Example.from_dict(nlp.make_doc(text), {"cats": cats})
        for text, cats in dev_docs
    ]

    best_score = 0.0
    best_nlp = None

    for epoch in range(TRAIN_EPOCHS):
        random.shuffle(train_examples)
        losses = {}
        batches = minibatch(train_examples, size=compounding(BATCH_SIZE_START, BATCH_SIZE_END, 1.001))
        for batch in batches:
            nlp.update(batch, drop=DROP, losses=losses, sgd=optimizer)

        # Evaluate on dev
        scores = nlp.evaluate(dev_examples)
        cats_score = scores.get("cats_macro_auc", 0.0)
        if cats_score > best_score:
            best_score = cats_score
            best_nlp = nlp.to_bytes()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{TRAIN_EPOCHS}: loss={losses.get('textcat_multilabel', 0):.4f}, "
                  f"dev_auc={cats_score:.4f}")

    print(f"    Best dev AUC: {best_score:.4f}")

    # Restore best
    nlp.from_bytes(best_nlp)
    return nlp


def predict_multilabel(nlp, text: str, all_labels: list[str], threshold: float = 0.5) -> list[str]:
    doc = nlp(text)
    active = [l for l, s in doc.cats.items() if s >= threshold]
    if not active:
        active = [max(doc.cats, key=doc.cats.get)]
    return active


def predict_single(nlp, text: str, all_labels: list[str]) -> str:
    doc = nlp(text)
    return max(doc.cats, key=doc.cats.get)


def main():
    examples = load_examples()
    train_ex, val_ex, test_ex = split_examples(examples)

    train_val = train_ex + val_ex  # use combined for final training

    print(f"Train+Val: {len(train_val)}, Test: {len(test_ex)}")

    # -----------------------------------------------------------------------
    # Train: urgency
    # -----------------------------------------------------------------------
    print("\nTraining urgency classifier...")
    urgency_train = [
        (ex["message"], make_cats([ex["urgency"]], URGENCY_LABELS))
        for ex in train_val
    ]
    urgency_dev = [
        (ex["message"], make_cats([ex["urgency"]], URGENCY_LABELS))
        for ex in val_ex
    ]
    urgency_nlp = build_and_train(urgency_train, urgency_dev, URGENCY_LABELS, "urgency")

    # -----------------------------------------------------------------------
    # Train: sentiment
    # -----------------------------------------------------------------------
    print("\nTraining sentiment classifier...")
    sentiment_train = [
        (ex["message"], make_cats([ex["sentiment"]], SENTIMENT_LABELS))
        for ex in train_val
    ]
    sentiment_dev = [
        (ex["message"], make_cats([ex["sentiment"]], SENTIMENT_LABELS))
        for ex in val_ex
    ]
    sentiment_nlp = build_and_train(sentiment_train, sentiment_dev, SENTIMENT_LABELS, "sentiment")

    # -----------------------------------------------------------------------
    # Train: categories (multi-label)
    # -----------------------------------------------------------------------
    print("\nTraining categories classifier...")
    cats_train = [
        (ex["message"], make_cats(ex["categories"], CATEGORIES))
        for ex in train_val
    ]
    cats_dev = [
        (ex["message"], make_cats(ex["categories"], CATEGORIES))
        for ex in val_ex
    ]
    cats_nlp = build_and_train(cats_train, cats_dev, CATEGORIES, "categories")

    # -----------------------------------------------------------------------
    # Evaluate on test set
    # -----------------------------------------------------------------------
    print("\nEvaluating on test set...")

    def predict_fn(message: str) -> dict:
        return {
            "urgency": predict_single(urgency_nlp, message, URGENCY_LABELS),
            "sentiment": predict_single(sentiment_nlp, message, SENTIMENT_LABELS),
            "categories": predict_multilabel(cats_nlp, message, CATEGORIES),
        }

    results = evaluate(test_ex, predict_fn)
    print_results("spaCy textcat (standard)", results)

    # Save
    output = {
        "approach": "spacy_textcat_standard",
        "train_epochs": TRAIN_EPOCHS,
        "dropout": DROP,
        "n_train_val": len(train_val),
        "n_test": len(test_ex),
        **results,
    }
    out_path = Path("data/results/04_spacy_textcat.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
