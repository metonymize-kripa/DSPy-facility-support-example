# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "pandas", "matplotlib", "seaborn"]
# ///
"""
00_data_exploration/explore.py
-------------------------------
Full EDA of the facility support dataset.

Sections:
  1. Label distributions
  2. Category co-occurrence
  3. Message length
  4. Vocabulary per label (top unigrams)
  5. N-gram analysis (bigrams/trigrams most distinctive per label)
  6. Hard cases
  7. Save results JSON

Run:
    uv run --group data 00_data_exploration/explore.py
    uv run --group data 00_data_exploration/explore.py --save-figures
    uv run --group data 00_data_exploration/explore.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dataset import (
    CATEGORIES, SENTIMENT_LABELS, URGENCY_LABELS,
    active_categories, load_examples, split_examples,
)

CAT_SHORT = {
    "routine_maintenance_requests":               "Routine maintenance",
    "customer_feedback_and_complaints":           "Feedback / complaints",
    "training_and_support_requests":              "Training & support",
    "quality_and_safety_concerns":                "Quality & safety",
    "sustainability_and_environmental_practices": "Sustainability",
    "cleaning_services_scheduling":               "Cleaning scheduling",
    "specialized_cleaning_services":              "Specialized cleaning",
    "emergency_repair_services":                  "Emergency repair",
    "facility_management_issues":                 "Facility management",
    "general_inquiries":                          "General inquiries",
}

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "i","you","we","they","he","she","it","my","your","our","their","its",
    "this","that","these","those","am","not","no","so","as","if","by","from",
    "up","about","into","than","then","them","there","here","when","where",
    "who","which","what","how","all","also","just","more","some","any","each",
    "please","thank","hope","message","finds","well","name","dear",
    "hi","hello","regards","best","sincerely","team","support","procare",
    "facility","solutions","services","would","like","appreciate","great",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-figures", action="store_true",
                   help="Save plots to 00_data_exploration/figures/")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib entirely")
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'\b[a-z]{3,}\b', text.lower())
            if w.lower() not in STOPWORDS]


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# 1. Label distributions
# ---------------------------------------------------------------------------

def label_distributions(examples: list[dict]) -> dict:
    n = len(examples)
    section("Label Distributions")

    urgency_counts = Counter(e["urgency"] for e in examples)
    sentiment_counts = Counter(e["sentiment"] for e in examples)

    print("\nUrgency:")
    for label in URGENCY_LABELS:
        c = urgency_counts.get(label, 0)
        bar = "█" * int(c / n * 40)
        print(f"  {label:8s} {c:3d} ({c/n:.1%})  {bar}")

    print("\nSentiment:")
    for label in SENTIMENT_LABELS:
        c = sentiment_counts.get(label, 0)
        bar = "█" * int(c / n * 40)
        print(f"  {label:10s} {c:3d} ({c/n:.1%})  {bar}")

    print("\nCategories (active frequency):")
    cat_counts = {}
    for cat in CATEGORIES:
        c = sum(1 for e in examples if e["categories"].get(cat, False))
        cat_counts[cat] = c
        bar = "█" * int(c / n * 40)
        print(f"  {cat:48s} {c:3d} ({c/n:.1%})  {bar}")

    n_active = [sum(e["categories"].values()) for e in examples]
    print(f"\nCategories per message:")
    for k in sorted(set(n_active)):
        c = n_active.count(k)
        print(f"  {k} active labels: {c} messages ({c/n:.1%})")

    return {
        "n": n,
        "urgency": dict(urgency_counts),
        "sentiment": dict(sentiment_counts),
        "categories": cat_counts,
    }


# ---------------------------------------------------------------------------
# 2. Category co-occurrence
# ---------------------------------------------------------------------------

def cooccurrence_matrix(examples: list[dict]) -> None:
    section("Category Co-occurrence (top pairs)")

    pairs: Counter = Counter()
    for e in examples:
        active = active_categories(e)
        for i, a in enumerate(active):
            for b in active[i+1:]:
                key = tuple(sorted([a, b]))
                pairs[key] += 1

    print("\nTop 15 co-occurring category pairs:")
    for (a, b), count in pairs.most_common(15):
        print(f"  {count:3d}x  {a.replace('_',' ')}  +  {b.replace('_',' ')}")

    print("\nCategories with zero or one co-occurrence partner:")
    solo = [c for c in CATEGORIES
            if not any(c in pair for pair in pairs if pairs[pair] > 1)]
    for c in solo:
        print(f"  {c}")


# ---------------------------------------------------------------------------
# 3. Message length
# ---------------------------------------------------------------------------

def message_length_analysis(examples: list[dict]) -> None:
    section("Message Length")

    lengths_words = [len(e["message"].split()) for e in examples]
    lengths_chars = [len(e["message"]) for e in examples]

    def stats(vals):
        s = sorted(vals)
        n = len(s)
        return {"min": s[0], "p25": s[n//4], "median": s[n//2],
                "p75": s[3*n//4], "max": s[-1], "mean": sum(vals)/n}

    ws = stats(lengths_words)
    cs = stats(lengths_chars)
    print(f"\nWord count:  min={ws['min']}  p25={ws['p25']}  median={ws['median']}"
          f"  p75={ws['p75']}  max={ws['max']}  mean={ws['mean']:.0f}")
    print(f"Char count:  min={cs['min']}  p25={cs['p25']}  median={cs['median']}"
          f"  p75={cs['p75']}  max={cs['max']}  mean={cs['mean']:.0f}")

    print("\nMedian word count by urgency:")
    for label in URGENCY_LABELS:
        subset = [len(e["message"].split()) for e in examples if e["urgency"] == label]
        if subset:
            print(f"  {label:8s}: {sorted(subset)[len(subset)//2]} words")

    short = [e for e in examples if len(e["message"].split()) < 50]
    print(f"\nShort messages (< 50 words): {len(short)} ({len(short)/len(examples):.1%})")


# ---------------------------------------------------------------------------
# 4. Vocabulary per label (top unigrams)
# ---------------------------------------------------------------------------

def vocabulary_analysis(examples: list[dict]) -> None:
    section("Top Unigrams per Urgency Level")
    for label in URGENCY_LABELS:
        subset = [e for e in examples if e["urgency"] == label]
        counter: Counter = Counter()
        for e in subset:
            counter.update(tokenize(e["message"]))
        top = [w for w, _ in counter.most_common(15)]
        print(f"\n  urgency={label}: {' | '.join(top)}")

    section("Top Unigrams per Sentiment")
    for label in SENTIMENT_LABELS:
        subset = [e for e in examples if e["sentiment"] == label]
        counter = Counter()
        for e in subset:
            counter.update(tokenize(e["message"]))
        top = [w for w, _ in counter.most_common(15)]
        print(f"\n  sentiment={label}: {' | '.join(top)}")


# ---------------------------------------------------------------------------
# 5. N-gram analysis
# ---------------------------------------------------------------------------

def ngram_analysis(examples: list[dict]) -> dict:
    """
    For each label group, find bigrams and trigrams that are distinctively
    more frequent in that group than in the rest of the corpus.

    Distinctiveness = (freq_in_group / total_in_group) /
                      (freq_outside / total_outside + 1e-9)

    This is a pointwise ratio — rough but interpretable without scipy.
    """
    section("N-gram Analysis (distinctive bigrams/trigrams per label)")

    results = {}

    def distinctive_ngrams(in_group: list[dict], out_group: list[dict],
                            n: int, top_k: int = 8) -> list[str]:
        def count(exs):
            c: Counter = Counter()
            total = 0
            for e in exs:
                toks = tokenize(e["message"])
                grams = ngrams(toks, n)
                c.update(grams)
                total += max(len(grams), 1)
            return c, total

        in_c, in_total = count(in_group)
        out_c, out_total = count(out_group)

        scores = {}
        for gram, cnt in in_c.items():
            if cnt < 2:  # require at least 2 occurrences
                continue
            in_rate = cnt / in_total
            out_rate = out_c.get(gram, 0) / out_total
            scores[gram] = in_rate / (out_rate + 1e-9)

        top = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [" ".join(g) for g in top]

    # By urgency
    print("\n  Distinctive BIGRAMS by urgency:")
    urgency_ngrams = {}
    for label in URGENCY_LABELS:
        in_g  = [e for e in examples if e["urgency"] == label]
        out_g = [e for e in examples if e["urgency"] != label]
        bi = distinctive_ngrams(in_g, out_g, n=2)
        tri = distinctive_ngrams(in_g, out_g, n=3)
        urgency_ngrams[label] = {"bigrams": bi, "trigrams": tri}
        print(f"    {label:8s}: {' | '.join(bi)}")

    print("\n  Distinctive TRIGRAMS by urgency:")
    for label in URGENCY_LABELS:
        tri = urgency_ngrams[label]["trigrams"]
        print(f"    {label:8s}: {' | '.join(tri)}")

    # By sentiment
    print("\n  Distinctive BIGRAMS by sentiment:")
    sentiment_ngrams = {}
    for label in SENTIMENT_LABELS:
        in_g  = [e for e in examples if e["sentiment"] == label]
        out_g = [e for e in examples if e["sentiment"] != label]
        bi = distinctive_ngrams(in_g, out_g, n=2)
        tri = distinctive_ngrams(in_g, out_g, n=3)
        sentiment_ngrams[label] = {"bigrams": bi, "trigrams": tri}
        print(f"    {label:10s}: {' | '.join(bi)}")

    print("\n  Distinctive TRIGRAMS by sentiment:")
    for label in SENTIMENT_LABELS:
        tri = sentiment_ngrams[label]["trigrams"]
        print(f"    {label:10s}: {' | '.join(tri)}")

    # Key insight check: do bigrams help distinguish urgency=high?
    high_bi = urgency_ngrams.get("high", {}).get("bigrams", [])
    print(f"\n  → High-urgency bigrams: {high_bi}")
    print("    (If these are explicit alarm phrases, keyword rules cover most high-urgency cases.)")
    print("    (If vague/absent, domain-knowledge is needed — LLMs earn their cost here.)")

    results["urgency_ngrams"] = urgency_ngrams
    results["sentiment_ngrams"] = sentiment_ngrams
    return results


# ---------------------------------------------------------------------------
# 6. Hard cases
# ---------------------------------------------------------------------------

def hard_cases(examples: list[dict]) -> None:
    section("Hard Cases")

    multi = [e for e in examples if sum(e["categories"].values()) >= 3]
    print(f"\nMessages with 3+ active categories: {len(multi)}")
    for e in multi[:3]:
        cats = active_categories(e)
        print(f"  [{e['urgency']}/{e['sentiment']}] {cats}")
        print(f"  {e['message'][:120].strip()}...")

    short = [e for e in examples if len(e["message"].split()) < 40]
    print(f"\nVery short messages (< 40 words): {len(short)}")
    for e in short[:3]:
        print(f"  [{e['urgency']}/{e['sentiment']}] {e['message'][:120].strip()}")

    positive_openers = re.compile(
        r'\b(appreciate|wonderful|excellent|great|satisfied|commend|impressed)\b', re.I)
    masked = [e for e in examples
              if e["sentiment"] != "positive"
              and positive_openers.search(e["message"][:300])]
    print(f"\nPotential register-masked sentiment (positive opener, non-positive label): "
          f"{len(masked)}")
    for e in masked[:3]:
        print(f"  [{e['sentiment']}] {e['message'][:150].strip()}...")


# ---------------------------------------------------------------------------
# 7. Save results JSON
# ---------------------------------------------------------------------------

def save_results(stats: dict) -> None:
    out = Path(__file__).parent.parent / "data" / "results" / "00_data_exploration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("Loading dataset...")
    examples = load_examples()
    train, val, test = split_examples(examples)
    print(f"Total: {len(examples)}  train={len(train)}  val={len(val)}  test={len(test)}")

    stats = label_distributions(examples)
    cooccurrence_matrix(examples)
    message_length_analysis(examples)
    vocabulary_analysis(examples)
    ngram_stats = ngram_analysis(examples)
    hard_cases(examples)

    stats["ngrams"] = ngram_stats

    # Save BEFORE plots so Ctrl+C on a plot window doesn't lose results
    save_results(stats)

    if not args.no_plots:
        try:
            _make_plots(examples, args.save_figures)
        except KeyboardInterrupt:
            print("\n(Plot window closed — results already saved)")
        except ImportError:
            print("\n(Skipping plots — matplotlib/seaborn not available)")


def _make_plots(examples, save):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. Label distribution bar charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    u_counts = Counter(e["urgency"] for e in examples)
    axes[0].bar(URGENCY_LABELS, [u_counts.get(l, 0) for l in URGENCY_LABELS],
                color=["#4CAF50", "#FF9800", "#F44336"])
    axes[0].set_title("Urgency distribution")
    axes[0].set_ylabel("Count")

    s_counts = Counter(e["sentiment"] for e in examples)
    axes[1].bar(SENTIMENT_LABELS, [s_counts.get(l, 0) for l in SENTIMENT_LABELS],
                color=["#2196F3", "#9E9E9E", "#F44336"])
    axes[1].set_title("Sentiment distribution")

    cat_counts = [sum(1 for e in examples if e["categories"].get(c, False))
                  for c in CATEGORIES]
    short_names = [CAT_SHORT.get(c, c) for c in CATEGORIES]
    axes[2].barh(short_names, cat_counts, color="#7986CB")
    axes[2].tick_params(axis="y", labelsize=9)
    axes[2].set_title("Category frequency")
    axes[2].set_xlabel("Count")

    plt.tight_layout()
    if save:
        plt.savefig(fig_dir / "label_distributions.png", dpi=150, bbox_inches="tight")
        print(f"Saved {fig_dir / 'label_distributions.png'}")
    else:
        plt.show()
    plt.close()

    # 2. Co-occurrence heatmap
    n = len(CATEGORIES)
    matrix = np.zeros((n, n))
    for e in examples:
        active = [i for i, c in enumerate(CATEGORIES) if e["categories"].get(c, False)]
        for i in active:
            for j in active:
                matrix[i][j] += 1

    fig, ax = plt.subplots(figsize=(11, 9))
    hm_labels = [CAT_SHORT.get(c, c) for c in CATEGORIES]
    sns.heatmap(matrix, xticklabels=hm_labels, yticklabels=hm_labels,
                annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("Category co-occurrence matrix")
    ax.tick_params(axis="x", labelsize=8, rotation=35)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    plt.tight_layout()
    if save:
        plt.savefig(fig_dir / "cooccurrence.png", dpi=150, bbox_inches="tight")
        print(f"Saved {fig_dir / 'cooccurrence.png'}")
    else:
        plt.show()
    plt.close()

    # 3. Top distinctive bigrams per urgency level (horizontal bar)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Most distinctive bigrams per urgency level", fontsize=13, fontweight="bold")

    def get_top_bigrams(in_group, out_group, top_k=8):
        def count(exs):
            c: Counter = Counter()
            total = 0
            for e in exs:
                toks = tokenize(e["message"])
                grams = ngrams(toks, 2)
                c.update(grams)
                total += max(len(grams), 1)
            return c, total
        in_c, in_total = count(in_group)
        out_c, out_total = count(out_group)
        scores = {}
        for gram, cnt in in_c.items():
            if cnt < 2:
                continue
            scores[gram] = (cnt / in_total) / (out_c.get(gram, 0) / out_total + 1e-9)
        top = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [(" ".join(g), scores[g]) for g in top]

    colors = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}
    for ax, label in zip(axes, URGENCY_LABELS):
        in_g  = [e for e in examples if e["urgency"] == label]
        out_g = [e for e in examples if e["urgency"] != label]
        top = get_top_bigrams(in_g, out_g)
        if top:
            grams, scores = zip(*top)
            ax.barh(list(reversed(grams)), list(reversed(scores)),
                    color=colors[label], alpha=0.8)
        ax.set_title(f"urgency = {label}")
        ax.set_xlabel("Distinctiveness ratio")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(fig_dir / "ngram_urgency.png", dpi=150, bbox_inches="tight")
        print(f"Saved {fig_dir / 'ngram_urgency.png'}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
