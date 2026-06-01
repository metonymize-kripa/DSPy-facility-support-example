# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
02_fasttext/prepare_data.py
----------------------------
Convert the dataset into FastText training format.

FastText format:
    __label__<label> <text>

For multi-label (categories):
    __label__cat1 __label__cat2 <text>

Run:
    uv run 02_fasttext/prepare_data.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import CATEGORIES, active_categories, load_examples, split_examples

OUT_DIR = Path(__file__).parent / "data"


def clean_text(text: str) -> str:
    """Normalise whitespace and remove newlines for FastText."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def write_fasttext(path: Path, examples: list[dict], task: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            text = clean_text(ex["message"])
            if task == "urgency":
                label = f"__label__{ex['urgency']}"
                f.write(f"{label} {text}\n")
            elif task == "sentiment":
                label = f"__label__{ex['sentiment']}"
                f.write(f"{label} {text}\n")
            elif task == "categories":
                active = active_categories(ex)
                if not active:
                    active = ["general_inquiries"]
                labels = " ".join(f"__label__{c}" for c in active)
                f.write(f"{labels} {text}\n")
    print(f"  Written {len(examples)} examples → {path}")


def main():
    print("Loading dataset...")
    examples = load_examples()
    train, val, test = split_examples(examples)

    # For FastText: train on train+val, evaluate on test
    trainval = train + val
    print(f"Train+val: {len(trainval)}  Test: {len(test)}")

    for task in ["urgency", "sentiment", "categories"]:
        print(f"\n{task}:")
        write_fasttext(OUT_DIR / f"{task}_train.txt", trainval, task)
        write_fasttext(OUT_DIR / f"{task}_test.txt", test, task)

    print("\nDone. Run train.py next.")


if __name__ == "__main__":
    main()
