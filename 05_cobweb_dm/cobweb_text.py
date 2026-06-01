"""
05_cobweb_dm/cobweb_text.py

Cobweb DM with TF-IDF features.

cobweb_keywords.py uses binary keyword features (hand-crafted vocabulary).
This script uses TF-IDF features (top-N terms by variance) instead — data-driven,
no domain knowledge injected.

Comparing purity between the two isolates the contribution of domain vocabulary:
  - Higher purity with keywords → domain terms are genuinely discriminative signals
  - Similar purity with TF-IDF → the signal is recoverable from raw term frequencies

Run:
    uv run --group cobweb 05_cobweb_dm/cobweb_text.py

Results saved to:
    data/results/05_cobweb_text.json
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["concept-formation>=1.0", "scikit-learn>=1.4"]
# ///

import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from concept_formation.cobweb import CobwebTree

from sklearn.feature_extraction.text import TfidfVectorizer

from shared.dataset import load_examples

N_FEATURES = 100  # top TF-IDF terms by variance to use as Cobweb features


def tfidf_to_cobweb_instance(feature_names: list[str], tfidf_row) -> dict:
    """Convert a sparse TF-IDF row to a Cobweb instance dict.
    
    Cobweb expects a dict of {feature: value}. We binarize at a threshold to
    keep the feature space manageable (Cobweb is sensitive to feature count).
    """
    arr = tfidf_row.toarray()[0]
    instance = {}
    for i, val in enumerate(arr):
        # Binarize: present if TF-IDF weight above mean non-zero
        instance[feature_names[i]] = "present" if val > 0.05 else "absent"
    return instance


def cluster_purity(tree, instances_with_labels, label_key: str) -> dict:
    """
    For each leaf concept in the tree, find the most common label.
    Purity = fraction of instances in that concept that have the majority label.
    
    Returns dict with per-concept purities and summary stats.
    """
    # Map each instance back to its concept via tree.categorize
    concept_labels = defaultdict(list)
    for inst, labels in instances_with_labels:
        concept = tree.categorize(inst)
        concept_id = id(concept)
        if isinstance(labels, list):
            concept_labels[concept_id].extend(labels)
        else:
            concept_labels[concept_id].append(labels)

    purities = []
    high_purity_count = 0
    for concept_id, label_list in concept_labels.items():
        if not label_list:
            continue
        counts = Counter(label_list)
        majority_count = counts.most_common(1)[0][1]
        purity = majority_count / len(label_list)
        purities.append(purity)
        if purity >= 0.8:
            high_purity_count += 1

    return {
        "n_concepts": len(purities),
        "mean_purity": round(sum(purities) / len(purities), 4) if purities else 0.0,
        "max_purity": round(max(purities), 4) if purities else 0.0,
        "high_purity_concepts": high_purity_count,
        "high_purity_fraction": round(high_purity_count / len(purities), 4) if purities else 0.0,
    }


def main():
    examples = load_examples()
    texts = [ex["message"] for ex in examples]

    print(f"Dataset: {len(examples)} examples")
    print(f"TF-IDF features: top {N_FEATURES} terms by variance")

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=N_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Top features: {feature_names[:20]}")

    # Convert to Cobweb instances
    instances = []
    for i in range(X.shape[0]):
        inst = tfidf_to_cobweb_instance(feature_names, X[i])
        inst["__idx__"] = i
        instances.append(inst)

    # Build tree
    print("\nBuilding Cobweb tree (TF-IDF features)...")
    tree = CobwebTree()
    for inst in instances:
        inst_copy = {k: v for k, v in inst.items() if k != "__idx__"}
        tree.ifit(inst_copy)

    print(f"Tree built.")

    # Purity analysis
    print("\nMeasuring cluster purity...")

    # Build (instance, label) pairs for categorization
    inst_label_pairs_urgency = [
        (
            {k: v for k, v in inst.items() if k != "__idx__"},
            examples[inst["__idx__"]]["urgency"],
        )
        for inst in instances
    ]
    inst_label_pairs_sentiment = [
        (
            {k: v for k, v in inst.items() if k != "__idx__"},
            examples[inst["__idx__"]]["sentiment"],
        )
        for inst in instances
    ]
    inst_label_pairs_categories = [
        (
            {k: v for k, v in inst.items() if k != "__idx__"},
            examples[inst["__idx__"]]["categories"],
        )
        for inst in instances
    ]

    urgency_purity = cluster_purity(tree, inst_label_pairs_urgency, "urgency")
    sentiment_purity = cluster_purity(tree, inst_label_pairs_sentiment, "sentiment")
    categories_purity = cluster_purity(tree, inst_label_pairs_categories, "categories")

    print("\n" + "="*60)
    print("CLUSTER PURITY: TF-IDF Cobweb")
    print("="*60)
    for task, purity in [("urgency", urgency_purity), ("sentiment", sentiment_purity), ("categories", categories_purity)]:
        print(f"\n  {task}:")
        print(f"    n_concepts:            {purity['n_concepts']}")
        print(f"    mean_purity:           {purity['mean_purity']:.4f}")
        print(f"    max_purity:            {purity['max_purity']:.4f}")
        print(f"    high_purity_concepts:  {purity['high_purity_concepts']} ({purity['high_purity_fraction']:.1%})")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("  Compare these values to cobweb_keywords.py results.")
    print("  If keyword-feature purity >> TF-IDF purity:")
    print("    Domain vocabulary is the key discriminant.")
    print("  If similar:")
    print("    Raw term frequencies capture the same structure.")
    print("  Either way, purity << 1.0 indicates labels don't map")
    print("  cleanly to Cobweb's discovered concepts — the task")
    print("  requires supervised signal, not just unsupervised grouping.")

    # Save
    output = {
        "approach": "cobweb_tfidf",
        "n_features": N_FEATURES,
        "n_examples": len(examples),
        "top_features": feature_names[:30],
        "urgency_purity": urgency_purity,
        "sentiment_purity": sentiment_purity,
        "categories_purity": categories_purity,
    }
    out_path = Path("data/results/05_cobweb_text.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
