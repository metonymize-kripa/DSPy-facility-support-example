"""
shared/dataset.py
-----------------
Single source of truth for dataset loading, parsing, and train/val/test splitting.
All approach scripts import from here so the data pipeline is identical across experiments.

Usage:
    from shared.dataset import load_examples, split_examples, CATEGORIES

    examples = load_examples()          # list of dicts
    train, val, test = split_examples(examples, seed=42)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import requests

DATASET_URL = (
    "https://raw.githubusercontent.com/meta-llama/prompt-ops/main/"
    "use-cases/facility-support-analyzer/dataset.json"
)

CACHE_PATH = Path(__file__).parent.parent / "data" / "dataset.json"

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


def load_examples(url: str = DATASET_URL, cache: bool = True) -> list[dict[str, Any]]:
    """
    Download (or load from cache) the dataset and return a list of parsed examples.

    Each example is a plain dict:
        {
            "message":    str,                  # raw email text
            "urgency":    "low"|"medium"|"high",
            "sentiment":  "positive"|"neutral"|"negative",
            "categories": dict[str, bool],      # keyed by CATEGORIES, value True/False
        }
    """
    raw = _fetch_raw(url, cache)
    examples = []
    for i, row in enumerate(raw):
        try:
            examples.append(_parse_row(i, row))
        except Exception as exc:
            print(f"WARNING: skipping row {i}: {exc}")
    return examples


def split_examples(
    examples: list[dict],
    seed: int = 42,
    train_frac: float = 0.33,
    val_frac: float = 0.33,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle and split into train / val / test."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    return shuffled[:t], shuffled[t:v], shuffled[v:]


def active_categories(example: dict) -> list[str]:
    """Return category labels that are True for this example."""
    return [c for c in CATEGORIES if example["categories"].get(c, False)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_raw(url: str, cache: bool) -> list[dict]:
    if cache and CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)

    print(f"Downloading dataset from {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(data, f)
        print(f"Cached to {CACHE_PATH}")

    return data


def _parse_row(i: int, row: dict) -> dict:
    message = (
        row.get("fields", {}).get("input")
        or row.get("message")
        or row.get("input")
        or ""
    ).strip()

    if not message:
        raise ValueError(f"no message text found (keys: {list(row.keys())})")

    answer_raw = row.get("answer", {})
    answer = json.loads(answer_raw) if isinstance(answer_raw, str) else answer_raw

    urgency = _normalize(answer.get("urgency", ""), URGENCY_LABELS)
    sentiment = _normalize(answer.get("sentiment", ""), SENTIMENT_LABELS)

    cats_raw = answer.get("categories", {})
    if isinstance(cats_raw, dict):
        categories = {c: bool(cats_raw.get(c, False)) for c in CATEGORIES}
    elif isinstance(cats_raw, list):
        active = set(cats_raw)
        categories = {c: c in active for c in CATEGORIES}
    else:
        raise ValueError(f"unexpected categories type: {type(cats_raw)}")

    return {
        "message": message,
        "urgency": urgency,
        "sentiment": sentiment,
        "categories": categories,
    }


def _normalize(value: str, allowed: list[str]) -> str:
    v = str(value).strip().lower()
    if v in allowed:
        return v
    for label in allowed:
        if re.search(rf"\b{re.escape(label)}\b", v):
            return label
    return v
