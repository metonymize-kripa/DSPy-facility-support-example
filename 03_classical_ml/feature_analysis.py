# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "scikit-learn", "numpy"]
# ///
"""
03_classical_ml/feature_analysis.py
-------------------------------------
Fit LogReg on train+val, print top TF-IDF features per label.
Validates (or contradicts) the handwritten rule vocabularies.

Run:
    uv run --group classical 03_classical_ml/feature_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import CATEGORIES, URGENCY_LABELS, SENTIMENT_LABELS, load_examples, split_examples


def top_features(pipeline, label_names: list[str], n: int = 15) -> None:
    from sklearn.pipeline import Pipeline
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    if hasattr(clf, "coef_"):
        coef = clf.coef_
    elif hasattr(clf, "estimators_"):
        # MultiOutputClassifier
        for i, estimator in enumerate(clf.estimators_):
            if hasattr(estimator, "coef_"):
                w = estimator.coef_[0]
                top_pos = [feature_names[j] for j in w.argsort()[-n:][::-1]]
                top_neg = [feature_names[j] for j in w.argsort()[:n]]
                cat = label_names[i]
                print(f"\n  {cat}:")
                print(f"    → {' | '.join(top_pos)}")
                print(f"    ← {' | '.join(top_neg)}")
        return

    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    for i, label in enumerate(label_names):
        w = coef[i]
        top_pos = [feature_names[j] for j in w.argsort()[-n:][::-1]]
        top_neg = [feature_names[j] for j in w.argsort()[:n]]
        print(f"\n  {label}:")
        print(f"    → {' | '.join(top_pos)}")
        print(f"    ← {' | '.join(top_neg)}")


def main():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    examples = load_examples()
    train, val, test = split_examples(examples)
    train_val = train + val

    def make_pipe(multioutput=False):
        base = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf = MultiOutputClassifier(base) if multioutput else base
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                                      sublinear_tf=True, min_df=1)),
            ("clf", clf),
        ])

    X = [e["message"] for e in train_val]

    print("\n" + "="*60)
    print("  Top features — URGENCY")
    print("="*60)
    pipe_u = make_pipe()
    pipe_u.fit(X, [e["urgency"] for e in train_val])
    top_features(pipe_u, URGENCY_LABELS)

    print("\n" + "="*60)
    print("  Top features — SENTIMENT")
    print("="*60)
    pipe_s = make_pipe()
    pipe_s.fit(X, [e["sentiment"] for e in train_val])
    top_features(pipe_s, SENTIMENT_LABELS)

    print("\n" + "="*60)
    print("  Top features — CATEGORIES (per label)")
    print("="*60)
    y_cat = [[int(e["categories"].get(c, False)) for c in CATEGORIES]
             for e in train_val]
    pipe_c = make_pipe(multioutput=True)
    pipe_c.fit(X, y_cat)
    top_features(pipe_c, CATEGORIES)


if __name__ == "__main__":
    main()
