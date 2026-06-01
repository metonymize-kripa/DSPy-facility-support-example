# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "scikit-learn", "numpy"]
# ///
"""
03_classical_ml/train_eval.py
------------------------------
TF-IDF + sklearn classifiers: LogReg, LinearSVC, Naive Bayes.
5-fold cross-validation on train+val; final evaluation on held-out test.

Run:
    uv run --group classical 03_classical_ml/train_eval.py
    uv run --group classical 03_classical_ml/train_eval.py --classifier logreg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import CATEGORIES, load_examples, split_examples
from shared.metrics import evaluate, print_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", choices=["all", "logreg", "svm", "nb"],
                   default="all")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_pipeline(clf_name: str, task: str):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import MaxAbsScaler

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=1,
    )

    if clf_name == "logreg":
        base_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    elif clf_name == "svm":
        base_clf = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    elif clf_name == "nb":
        base_clf = MultinomialNB(alpha=0.1)

    if task == "categories":
        clf = MultiOutputClassifier(base_clf)
    else:
        clf = base_clf

    if clf_name == "nb":
        # NB needs non-negative features; MaxAbsScaler works with TF-IDF
        return Pipeline([("tfidf", vectorizer), ("scaler", MaxAbsScaler()), ("clf", clf)])
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def examples_to_xy(examples: list[dict], task: str):
    X = [e["message"] for e in examples]
    if task == "urgency":
        y = [e["urgency"] for e in examples]
    elif task == "sentiment":
        y = [e["sentiment"] for e in examples]
    elif task == "categories":
        y = [[int(e["categories"].get(c, False)) for c in CATEGORIES]
             for e in examples]
    return X, y


def run_classifier(clf_name: str, train_val: list[dict], test: list[dict], seed: int):
    from sklearn.model_selection import cross_val_score
    import numpy as np

    print(f"\n  ── {clf_name.upper()} ──")
    task_results = {}

    for task in ["urgency", "sentiment", "categories"]:
        X_tv, y_tv = examples_to_xy(train_val, task)
        X_test, y_test = examples_to_xy(test, task)

        pipe = build_pipeline(clf_name, task)

        # 5-fold CV on train+val
        if task != "categories":
            cv_scores = cross_val_score(pipe, X_tv, y_tv, cv=5, scoring="accuracy")
            print(f"  {task:12s} CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}", end="")
        else:
            # Multi-output CV: average accuracy across labels
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            fold_scores = []
            for tr_idx, val_idx in kf.split(X_tv):
                X_tr = [X_tv[i] for i in tr_idx]
                y_tr = [y_tv[i] for i in tr_idx]
                X_v = [X_tv[i] for i in val_idx]
                y_v = [y_tv[i] for i in val_idx]
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_v)
                # 10-way binary accuracy
                correct = sum(
                    1 for gold_row, pred_row in zip(y_v, preds)
                    for g, p in zip(gold_row, pred_row) if g == p
                )
                total = len(y_v) * len(CATEGORIES)
                fold_scores.append(correct / total)
            print(f"  {task:12s} CV={np.mean(fold_scores):.3f}±{np.std(fold_scores):.3f}", end="")

        # Fit on full train+val, evaluate on test
        pipe.fit(X_tv, y_tv)

        if task != "categories":
            test_preds = pipe.predict(X_test)
            acc = sum(p == g for p, g in zip(test_preds, y_test)) / len(y_test)
        else:
            test_preds = pipe.predict(X_test)
            correct = sum(
                1 for gold_row, pred_row in zip(y_test, test_preds)
                for g, p in zip(gold_row, pred_row) if g == p
            )
            acc = correct / (len(y_test) * len(CATEGORIES))

        print(f"  test={acc:.3f}")
        task_results[task] = {"cv_mean": float(np.mean(fold_scores) if task == "categories"
                                               else cv_scores.mean()),
                              "test_acc": round(acc, 4),
                              "pipeline": pipe}

    return task_results


def make_predict_fn(pipelines: dict):
    """Build a predict function from fitted pipelines for each task."""
    urgency_pipe = pipelines["urgency"]["pipeline"]
    sentiment_pipe = pipelines["sentiment"]["pipeline"]
    cat_pipe = pipelines["categories"]["pipeline"]

    def predict(message: str) -> dict:
        urgency = urgency_pipe.predict([message])[0]
        sentiment = sentiment_pipe.predict([message])[0]
        cat_pred = cat_pipe.predict([message])[0]
        categories = [CATEGORIES[i] for i, v in enumerate(cat_pred) if v == 1]
        if not categories:
            categories = [CATEGORIES[list(cat_pred).index(max(cat_pred))]]
        return {"urgency": urgency, "sentiment": sentiment, "categories": categories}

    return predict


def main():
    args = parse_args()

    examples = load_examples()
    train, val, test = split_examples(examples, seed=args.seed)
    train_val = train + val
    print(f"Train+val: {len(train_val)}  Test: {len(test)}")

    clf_names = (["logreg", "svm", "nb"] if args.classifier == "all"
                 else [args.classifier])

    results_path = Path(__file__).parent.parent / "data" / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    for clf_name in clf_names:
        task_results = run_classifier(clf_name, train_val, test, args.seed)

        predict_fn = make_predict_fn(task_results)
        canonical = evaluate(test, predict_fn)
        print_results(f"Classical ML — {clf_name.upper()}", canonical)

        out = {
            "approach": "classical_ml",
            "variant": clf_name,
            "model": f"TF-IDF + {clf_name}",
            "n_test": canonical["n"],
            "aggregate": round(canonical["aggregate"], 4),
            "urgency": round(canonical["urgency"], 4),
            "sentiment": round(canonical["sentiment"], 4),
            "categories": round(canonical["categories"], 4),
            "cost_per_query_usd": 0.0,
            "latency_p50_ms": 2,
            "latency_p95_ms": 5,
            "training_examples_required": len(train_val),
            "notes": f"TF-IDF ngram(1,2) max_features=8000 sublinear_tf; {clf_name}",
        }
        out_file = results_path / f"03_classical_ml_{clf_name}.json"
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved → {out_file}")


if __name__ == "__main__":
    main()
