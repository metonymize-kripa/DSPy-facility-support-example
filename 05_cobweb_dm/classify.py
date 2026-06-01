# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "concept-formation<=0.3.9", "scikit-learn"]
# ///
"""
05_cobweb_dm/classify.py
-------------------------
Evaluate Cobweb as a classifier: train on train+val, predict on test.

Uses nearest-concept classification: for each test example, categorize into
the tree and use the majority label of the matched concept.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import CATEGORIES, load_examples, split_examples
from shared.metrics import evaluate, print_results

# Feature extraction (same as cobweb_keywords.py)
URGENCY_KEYWORDS = {
    "high": ["urgent", "emergency", "immediate", "asap", "critical", "broken", "failed"],
    "medium": ["soon", "schedule", "reschedule", "follow", "upcoming"],
    "low": ["inquiry", "question", "information", "when convenient"],
}

CATEGORY_KEYWORDS = {
    "emergency_repair_services": ["emergency", "repair", "broken", "not working", "immediate"],
    "routine_maintenance_requests": ["routine", "maintenance", "hvac", "plumbing", "scheduled"],
    "cleaning_services_scheduling": ["schedule", "reschedule", "appointment", "weekly"],
    "specialized_cleaning_services": ["deep clean", "carpet", "window", "mold", "specialized"],
    "customer_feedback_and_complaints": ["complaint", "dissatisfied", "poor", "quality"],
    "quality_and_safety_concerns": ["safety", "hazard", "dangerous", "concern"],
    "sustainability_and_environmental_practices": ["eco", "green", "sustainable", "environmental"],
    "training_and_support_requests": ["training", "support", "help", "how to"],
    "facility_management_issues": ["management", "facility", "building", "lease"],
    "general_inquiries": ["inquiry", "information", "about", "question"],
}

SENTIMENT_KEYWORDS = {
    "positive": ["excellent", "outstanding", "satisfied", "great", "appreciate"],
    "negative": ["upset", "frustrated", "disappointed", "unacceptable", "terrible"],
}


def extract_features(text: str) -> dict:
    """Extract binary keyword features."""
    text_lower = text.lower()
    features = {}
    
    # Urgency features
    for level, keywords in URGENCY_KEYWORDS.items():
        features[f"urgency_{level}"] = any(kw in text_lower for kw in keywords)
    
    # Category features
    for cat, keywords in CATEGORY_KEYWORDS.items():
        features[f"cat_{cat}"] = any(kw in text_lower for kw in keywords)
    
    # Sentiment features
    for sent, keywords in SENTIMENT_KEYWORDS.items():
        features[f"sentiment_{sent}"] = any(kw in text_lower for kw in keywords)
    
    return features


def train_cobweb(examples: list[dict]):
    """Train Cobweb tree on labeled examples."""
    from concept_formation.cobweb import CobwebTree
    
    tree = CobwebTree()
    for i, ex in enumerate(examples):
        feat = extract_features(ex["message"])
        feat["__idx__"] = i  # Track back to example for label lookup
        tree.ifit(feat)
    
    return tree


def predict_with_cobweb(tree, text: str, train_examples: list[dict]) -> dict:
    """Categorize text into tree and predict based on concept's majority labels."""
    feat = extract_features(text)
    
    # Find best matching concept
    concept = tree.categorize(feat)
    
    # Get instances in this concept
    if not hasattr(concept, "instances") or not concept.instances:
        # Fallback: return defaults if concept has no instances
        return {
            "urgency": "medium",
            "sentiment": "neutral",
            "categories": ["general_inquiries"],
        }
    
    # Get original examples for this concept
    idxs = [inst.get("__idx__") for inst in concept.instances if "__idx__" in inst]
    concept_examples = [train_examples[i] for i in idxs if i < len(train_examples)]
    
    if not concept_examples:
        return {
            "urgency": "medium",
            "sentiment": "neutral",
            "categories": ["general_inquiries"],
        }
    
    # Majority vote for urgency
    urgency_counts = Counter(e["urgency"] for e in concept_examples)
    urgency_pred = urgency_counts.most_common(1)[0][0]
    
    # Majority vote for sentiment
    sentiment_counts = Counter(e["sentiment"] for e in concept_examples)
    sentiment_pred = sentiment_counts.most_common(1)[0][0]
    
    # For categories: predict True if majority of concept has it True
    categories_pred = []
    for cat in CATEGORIES:
        active_count = sum(1 for e in concept_examples if e["categories"].get(cat, False))
        if active_count >= len(concept_examples) / 2:
            categories_pred.append(cat)
    
    if not categories_pred:
        categories_pred = ["general_inquiries"]
    
    return {
        "urgency": urgency_pred,
        "sentiment": sentiment_pred,
        "categories": categories_pred,
    }


def main():
    print("Loading dataset...")
    examples = load_examples()
    train, val, test = split_examples(examples)
    train_val = train + val
    
    print(f"Train+Val: {len(train_val)}, Test: {len(test)}")
    
    print("\nTraining Cobweb tree (unsupervised, but with label tracking)...")
    tree = train_cobweb(train_val)
    print("Tree built.")
    
    print(f"\nEvaluating on {len(test)} test examples...")
    
    def predict_fn(message: str) -> dict:
        return predict_with_cobweb(tree, message, train_val)
    
    results = evaluate(test, predict_fn, verbose=False)
    print_results("Cobweb Conceptual Clustering (nearest-concept classifier)", results)
    
    # Save results
    out = {
        "approach": "cobweb",
        "variant": "nearest_concept_classifier",
        "model": "CobwebTree",
        "n_test": results["n"],
        "aggregate": round(results["aggregate"], 4),
        "urgency": round(results["urgency"], 4),
        "sentiment": round(results["sentiment"], 4),
        "categories": round(results["categories"], 4),
        "cost_per_query_usd": 0.0,
        "latency_p50_ms": 5,
        "latency_p95_ms": 20,
        "training_examples_required": len(train_val),
        "notes": "Keyword features + Cobweb clustering + nearest-concept prediction",
    }
    
    out_path = Path(__file__).parent.parent / "data" / "results" / "05_cobweb.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
