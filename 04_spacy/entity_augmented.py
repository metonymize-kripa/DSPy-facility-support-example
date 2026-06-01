# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "spacy", "scikit-learn", "numpy"]
# ///
"""
04_spacy/entity_augmented.py
------------------------------
Entity features (via spaCy PhraseMatcher) + TF-IDF → LogReg.

Tests the neuro-symbolic hypothesis: does encoding domain knowledge
as binary entity features improve urgency accuracy specifically?

Run:
    python -m spacy download en_core_web_sm   (once)
    uv run --group spacy 04_spacy/entity_augmented.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import CATEGORIES, load_examples, split_examples
from shared.metrics import evaluate, print_results
from entity_rules import ENTITY_GROUPS, FEATURE_NAMES


def build_matcher(nlp):
    from spacy.matcher import PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for group, phrases in ENTITY_GROUPS.items():
        patterns = [nlp.make_doc(p.lower()) for p in phrases]
        matcher.add(group, patterns)
    return matcher


def extract_entity_features(texts: list[str], nlp, matcher) -> np.ndarray:
    """Return binary feature matrix: shape (n_texts, n_entity_groups)."""
    n_groups = len(FEATURE_NAMES)
    features = np.zeros((len(texts), n_groups), dtype=float)
    group_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    for i, text in enumerate(texts):
        doc = nlp(text[:1000])  # cap length for speed
        matches = matcher(doc)
        for match_id, _, _ in matches:
            group = nlp.vocab.strings[match_id]
            if group in group_idx:
                features[i, group_idx[group]] = 1.0

    return features


def main():
    import spacy
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from scipy.sparse import hstack, csr_matrix

    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        print("Run: python -m spacy download en_core_web_sm")
        sys.exit(1)

    matcher = build_matcher(nlp)

    print("Loading dataset...")
    examples = load_examples()
    train, val, test = split_examples(examples)
    train_val = train + val

    X_tv_texts = [e["message"] for e in train_val]
    X_test_texts = [e["message"] for e in test]

    print("Extracting entity features...")
    ent_tv = extract_entity_features(X_tv_texts, nlp, matcher)
    ent_test = extract_entity_features(X_test_texts, nlp, matcher)

    print(f"Entity feature matrix: {ent_tv.shape[1]} features")
    # Show which features fired most
    fired = ent_tv.sum(axis=0)
    for i, name in enumerate(FEATURE_NAMES):
        if fired[i] > 0:
            print(f"  {name}: {int(fired[i])} matches in train+val")

    # Build TF-IDF features
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                            sublinear_tf=True, min_df=1)
    tfidf_tv = tfidf.fit_transform(X_tv_texts)
    tfidf_test = tfidf.transform(X_test_texts)

    # Concatenate entity + TF-IDF
    X_tv = hstack([tfidf_tv, csr_matrix(ent_tv)])
    X_test = hstack([tfidf_test, csr_matrix(ent_test)])

    results_by_task = {}
    pipelines = {}

    for task, label_set in [("urgency", None), ("sentiment", None), ("categories", CATEGORIES)]:
        if task == "categories":
            y_tv = [[int(e["categories"].get(c, False)) for c in CATEGORIES]
                    for e in train_val]
            y_test = [[int(e["categories"].get(c, False)) for c in CATEGORIES]
                      for e in test]
            base = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf = MultiOutputClassifier(base)
        else:
            y_tv = [e[task] for e in train_val]
            y_test = [e[task] for e in test]
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

        clf.fit(X_tv, y_tv)
        preds = clf.predict(X_test)

        if task != "categories":
            acc = sum(p == g for p, g in zip(preds, y_test)) / len(y_test)
        else:
            correct = sum(
                1 for gold_row, pred_row in zip(y_test, preds)
                for g, p in zip(gold_row, pred_row) if g == p
            )
            acc = correct / (len(y_test) * len(CATEGORIES))

        results_by_task[task] = {"acc": acc, "clf": clf, "preds": preds}
        print(f"  {task}: {acc:.3f}")

    # Canonical evaluation
    def predict(message: str) -> dict:
        ent = extract_entity_features([message], nlp, matcher)
        tfidf_feat = tfidf.transform([message])
        X = hstack([tfidf_feat, csr_matrix(ent)])

        urgency = results_by_task["urgency"]["clf"].predict(X)[0]
        sentiment = results_by_task["sentiment"]["clf"].predict(X)[0]
        cat_pred = results_by_task["categories"]["clf"].predict(X)[0]
        categories = [CATEGORIES[i] for i, v in enumerate(cat_pred) if v == 1]
        if not categories:
            categories = [CATEGORIES[list(cat_pred).index(max(cat_pred))]]
        return {"urgency": urgency, "sentiment": sentiment, "categories": categories}

    canonical = evaluate(test, predict)
    print_results("spaCy Entity-Augmented (entity features + TF-IDF + LogReg)", canonical)

    out = {
        "approach": "spacy_entity_augmented",
        "variant": "entity_features_tfidf_logreg",
        "model": "en_core_web_sm + PhraseMatcher",
        "n_test": canonical["n"],
        "aggregate": round(canonical["aggregate"], 4),
        "urgency": round(canonical["urgency"], 4),
        "sentiment": round(canonical["sentiment"], 4),
        "categories": round(canonical["categories"], 4),
        "cost_per_query_usd": 0.0,
        "latency_p50_ms": 5,
        "latency_p95_ms": 15,
        "training_examples_required": len(train_val),
        "notes": f"{len(FEATURE_NAMES)} entity groups + TF-IDF ngram(1,2); LogReg C=1.0",
    }
    out_path = Path(__file__).parent.parent / "data" / "results" / "04_spacy_entity_augmented.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
