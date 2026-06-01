# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "concept-formation", "scikit-learn"]
# ///
"""
05_cobweb_dm/cobweb_keywords.py
--------------------------------
Cobweb conceptual clustering on interpretable binary keyword features.

No labels used for clustering — purely unsupervised.
After clustering, measure how well the discovered concepts align
with the gold urgency / category labels.

Run:
    uv run --group cobweb 05_cobweb_dm/cobweb_keywords.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import CATEGORIES, URGENCY_LABELS, SENTIMENT_LABELS, load_examples

# Reuse entity group vocabulary as binary features
sys.path.insert(0, str(Path(__file__).parent.parent / "04_spacy"))
from entity_rules import ENTITY_GROUPS, FEATURE_NAMES


def extract_binary_features(message: str) -> dict[str, str]:
    """
    Return a Cobweb-compatible nominal attribute dict.
    Each feature is "1" or "0" (Cobweb treats these as nominal values).
    """
    text_lower = message.lower()
    features = {}
    for group, phrases in ENTITY_GROUPS.items():
        present = any(p.lower() in text_lower for p in phrases)
        features[group] = "1" if present else "0"
    return features


def cluster_purity(node, label_fn, label_values) -> dict[str, float]:
    """
    Compute purity of a Cobweb node w.r.t. a label function.
    purity = (count of majority label) / total examples in node
    """
    instances = list(node.instances) if hasattr(node, "instances") else []
    if not instances:
        return {}

    label_counts = Counter(label_fn(inst) for inst in instances)
    majority = label_counts.most_common(1)[0][1]
    purity = majority / len(instances)
    return {
        "purity": round(purity, 3),
        "n": len(instances),
        "distribution": dict(label_counts),
    }


def evaluate_tree(tree, examples: list[dict]) -> None:
    """Walk the tree and report purity statistics."""
    from concept_formation.cobweb import CobwebNode

    # Collect all leaf nodes
    def get_leaves(node):
        if not node.children:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(get_leaves(child))
        return leaves

    # Map each instance back to its example using index
    # We store index in the instance dict
    leaves = get_leaves(tree.root)

    print(f"\nTotal concepts (leaf nodes): {len(leaves)}")
    print(f"Total internal nodes: {_count_internal(tree.root)}")

    # For each leaf, compute purity w.r.t. urgency
    urgency_purities = []
    sentiment_purities = []
    for leaf in leaves:
        if not hasattr(leaf, "instances") or not leaf.instances:
            continue
        # Get original examples by idx
        idxs = [inst.get("__idx__") for inst in leaf.instances if "__idx__" in inst]
        leaf_examples = [examples[i] for i in idxs if i < len(examples)]
        if not leaf_examples:
            continue

        u_counts = Counter(e["urgency"] for e in leaf_examples)
        s_counts = Counter(e["sentiment"] for e in leaf_examples)

        u_purity = u_counts.most_common(1)[0][1] / len(leaf_examples)
        s_purity = s_counts.most_common(1)[0][1] / len(leaf_examples)
        urgency_purities.append(u_purity)
        sentiment_purities.append(s_purity)

    if urgency_purities:
        print(f"\nUrgency purity across leaf concepts:")
        print(f"  Mean:   {sum(urgency_purities)/len(urgency_purities):.3f}")
        print(f"  Max:    {max(urgency_purities):.3f}")
        print(f"  >0.8:   {sum(1 for p in urgency_purities if p > 0.8)} concepts")

    if sentiment_purities:
        print(f"\nSentiment purity across leaf concepts:")
        print(f"  Mean:   {sum(sentiment_purities)/len(sentiment_purities):.3f}")
        print(f"  Max:    {max(sentiment_purities):.3f}")
        print(f"  >0.8:   {sum(1 for p in sentiment_purities if p > 0.8)} concepts")

    # Per-category purity
    print("\nPer-category purity (max across leaves):")
    for cat in CATEGORIES:
        purities = []
        for leaf in leaves:
            if not hasattr(leaf, "instances") or not leaf.instances:
                continue
            idxs = [inst.get("__idx__") for inst in leaf.instances if "__idx__" in inst]
            leaf_examples = [examples[i] for i in idxs if i < len(examples)]
            if not leaf_examples:
                continue
            active = sum(1 for e in leaf_examples if e["categories"].get(cat, False))
            total = len(leaf_examples)
            # purity for "true" vs "false"
            p = max(active, total - active) / total
            purities.append(p)
        if purities:
            print(f"  {cat:<48} max={max(purities):.3f}  mean={sum(purities)/len(purities):.3f}")


def _count_internal(node) -> int:
    if not node.children:
        return 0
    return 1 + sum(_count_internal(c) for c in node.children)


def main():
    try:
        from concept_formation.cobweb import CobwebTree
    except ImportError:
        print("Install: uv sync --group cobweb")
        sys.exit(1)

    print("Loading dataset (all 200 examples — unsupervised, no split)...")
    examples = load_examples()

    print("Extracting keyword features...")
    instances = []
    for i, ex in enumerate(examples):
        feat = extract_binary_features(ex["message"])
        feat["__idx__"] = i   # track back to original example
        instances.append(feat)

    print(f"Feature space: {len(FEATURE_NAMES)} binary attributes")
    print("Running Cobweb (incremental)...")

    tree = CobwebTree()
    for inst in instances:
        tree.ifit(inst)

    print(f"\nTree built. Root has {len(tree.root.children)} direct children.")

    evaluate_tree(tree, examples)

    # Save summary
    out = {
        "approach": "cobweb",
        "variant": "keyword_features",
        "model": "CobwebTree",
        "n_examples": len(examples),
        "n_features": len(FEATURE_NAMES),
        "notes": f"{len(FEATURE_NAMES)} binary entity-group features; no labels used",
    }
    out_path = Path(__file__).parent.parent / "data" / "results" / "05_cobweb_keywords.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
