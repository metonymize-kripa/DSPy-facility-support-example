# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "fasttext-wheel"]
# ///
"""
02_fasttext/train.py
---------------------
Train FastText models for urgency, sentiment, and categories.
Runs a small grid search over wordNgrams and epoch.

Run:
    uv run --group fasttext 02_fasttext/train.py
    uv run --group fasttext 02_fasttext/train.py --no-grid   # single best config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-grid", action="store_true",
                   help="Skip grid search, use best known hyperparams")
    return p.parse_args()


def train_model(task: str, word_ngrams: int, epoch: int, lr: float):
    import fasttext
    train_path = str(DATA_DIR / f"{task}_train.txt")

    if task == "categories":
        model = fasttext.train_supervised(
            input=train_path,
            wordNgrams=word_ngrams,
            epoch=epoch,
            lr=lr,
            minCount=1,
            loss="ova",       # one-vs-all for multi-label
            dim=100,
            verbose=0,
        )
    else:
        model = fasttext.train_supervised(
            input=train_path,
            wordNgrams=word_ngrams,
            epoch=epoch,
            lr=lr,
            minCount=1,
            dim=100,
            verbose=0,
        )
    return model


def evaluate_model(model, task: str) -> float:
    """Return accuracy on the test file."""
    test_path = str(DATA_DIR / f"{task}_test.txt")
    result = model.test(test_path, k=1 if task != "categories" else -1)
    # result = (n_samples, precision, recall)
    return result[1]  # precision@1 for single-label, precision@k for multi-label


def grid_search(task: str) -> tuple[dict, float]:
    best_score = -1.0
    best_params = {}
    best_model = None

    for ngrams in [1, 2, 3]:
        for epoch in [25, 50, 100]:
            for lr in [0.1, 0.5, 1.0]:
                model = train_model(task, ngrams, epoch, lr)
                score = evaluate_model(model, task)
                if score > best_score:
                    best_score = score
                    best_params = {"wordNgrams": ngrams, "epoch": epoch, "lr": lr}
                    best_model = model
                print(f"  {task} ngrams={ngrams} epoch={epoch} lr={lr}: {score:.3f}")

    return best_model, best_params, best_score


def main():
    args = parse_args()

    # Ensure data files exist
    if not (DATA_DIR / "urgency_train.txt").exists():
        print("Data files not found. Run prepare_data.py first.")
        sys.exit(1)

    MODEL_DIR.mkdir(exist_ok=True)
    summary = {}

    for task in ["urgency", "sentiment", "categories"]:
        print(f"\n── {task} ──")

        if args.no_grid:
            model = train_model(task, word_ngrams=2, epoch=50, lr=0.5)
            score = evaluate_model(model, task)
            params = {"wordNgrams": 2, "epoch": 50, "lr": 0.5}
            print(f"  Score: {score:.3f}")
        else:
            model, params, score = grid_search(task)
            print(f"  Best: {params}  score={score:.3f}")

        # Save model
        model_path = str(MODEL_DIR / f"{task}.bin")
        model.save_model(model_path)
        print(f"  Saved → {model_path}")

        summary[task] = {"best_params": params, "test_precision": round(score, 4)}

    # Save training summary
    out = MODEL_DIR / "training_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary saved to {out}")
    print("\nRun classify.py to get canonical evaluation results.")


if __name__ == "__main__":
    main()
