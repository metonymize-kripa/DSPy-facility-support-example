"""
07_evaluation/plot_tradeoffs.py

Generate the paper's main figure: a complexity/cost vs. accuracy tradeoff plot.

X-axis: complexity tier (1=rules, 2=FastText, 3=classical ML, 4=spaCy, 5=LLM zero-shot, 6=GEPA)
Y-axis: aggregate accuracy

Also generates:
  - Per-sub-task breakdown bar chart (urgency / sentiment / categories)
  - Knowledge-type heatmap showing where each approach gains/loses

Requires matplotlib. Run after all approach scripts have produced results.

Run:
    uv run --group eval 07_evaluation/plot_tradeoffs.py
    uv run --group eval 07_evaluation/plot_tradeoffs.py --save figures/

Figures saved to figures/ (or displayed interactively if --save not given).
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26"]
# ///

import sys
import json
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

APPROACHES = [
    # (label, filename, tier, color)
    ("Rules",              "01_handwritten_rules.json",       1, "#6c757d"),
    ("FastText",           "02_fasttext.json",                2, "#fd7e14"),
    ("Classical ML",       "03_classical_ml_best.json",       3, "#20c997"),
    ("spaCy entity",       "04_spacy_entity_augmented.json",  4, "#0dcaf0"),
    ("spaCy textcat",      "04_spacy_textcat.json",           4, "#0d6efd"),
    ("gemma4:26b 0-shot",  "06_llm_parent_base.json",         5, "#d63384"),
    ("gemma4:e4b 0-shot",  "06_llm_student_base.json",        5, "#6f42c1"),
    ("gemma4:e4b + GEPA",  "06_llm_student_compiled.json",    6, "#198754"),
]

TIER_LABELS = {
    1: "Handwritten\nRules",
    2: "Bag-of-words\n(FastText)",
    3: "Classical ML\n(TF-IDF+LR)",
    4: "spaCy\nPipeline",
    5: "LLM\nZero-shot",
    6: "LLM +\nPrompt Opt.",
}


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def plot_tradeoff(results, save_dir: Path | None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: aggregate tradeoff line ----
    ax = axes[0]
    ax.set_title("Complexity Ladder vs. Aggregate Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Approach tier (increasing complexity →)", fontsize=11)
    ax.set_ylabel("Aggregate accuracy", fontsize=11)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(list(TIER_LABELS.keys()))
    ax.set_xticklabels([TIER_LABELS[k] for k in TIER_LABELS], fontsize=8)
    ax.axhline(y=0.754, color="#adb5bd", linestyle="--", linewidth=1, label="DSPy tutorial baseline (0.754)")
    ax.grid(axis="y", alpha=0.3)

    for label, filename, tier, color in APPROACHES:
        r = results.get(filename)
        if r is None:
            continue
        agg = r.get("aggregate")
        if agg is None:
            continue
        ax.scatter(tier + np.random.uniform(-0.1, 0.1), agg, color=color, s=80, zorder=5, label=label)

    ax.legend(fontsize=8, loc="lower right")

    # ---- Right: per-task breakdown bar chart ----
    ax2 = axes[1]
    ax2.set_title("Per-Subtask Accuracy by Approach", fontsize=13, fontweight="bold")

    approach_labels = []
    urgency_vals = []
    sentiment_vals = []
    category_vals = []
    bar_colors = []

    for label, filename, tier, color in APPROACHES:
        r = results.get(filename)
        if r is None:
            continue
        if all(r.get(k) is not None for k in ["urgency", "sentiment", "categories"]):
            approach_labels.append(label)
            urgency_vals.append(r["urgency"])
            sentiment_vals.append(r["sentiment"])
            category_vals.append(r["categories"])
            bar_colors.append(color)

    if approach_labels:
        x = np.arange(len(approach_labels))
        width = 0.25
        ax2.bar(x - width, urgency_vals,   width, label="Urgency",    color="#6f42c1", alpha=0.8)
        ax2.bar(x,          sentiment_vals, width, label="Sentiment",  color="#0d6efd", alpha=0.8)
        ax2.bar(x + width, category_vals,  width, label="Categories", color="#198754", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(approach_labels, rotation=25, ha="right", fontsize=8)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.set_ylim(0.0, 1.0)
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "tradeoff_plot.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()

    plt.close()


def plot_knowledge_heatmap(results, save_dir: Path | None):
    """Heatmap: approach × knowledge_type, cell = relative performance."""
    # Knowledge type predictions from CONCEPTUAL_FRAMEWORK.md:
    # Type1=Taxonomy, Type2=Domain, Type3=Register
    # Proxy: categories score ~ Type1, urgency ~ Type2, sentiment ~ Type3

    approach_labels = []
    type1_vals = []  # categories ~ taxonomic knowledge
    type2_vals = []  # urgency    ~ domain knowledge
    type3_vals = []  # sentiment  ~ register knowledge

    for label, filename, tier, color in APPROACHES:
        r = results.get(filename)
        if r is None:
            continue
        if all(r.get(k) is not None for k in ["urgency", "sentiment", "categories"]):
            approach_labels.append(label)
            type1_vals.append(r["categories"])
            type2_vals.append(r["urgency"])
            type3_vals.append(r["sentiment"])

    if not approach_labels:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(
        "Knowledge Type Coverage by Approach\n"
        "(Type1=Taxonomy→Categories, Type2=Domain→Urgency, Type3=Register→Sentiment)",
        fontsize=11, fontweight="bold"
    )

    data = np.array([type1_vals, type2_vals, type3_vals])
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(approach_labels)))
    ax.set_xticklabels(approach_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Type 1\n(Taxonomy)", "Type 2\n(Domain)", "Type 3\n(Register)"], fontsize=10)

    for i in range(3):
        for j in range(len(approach_labels)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()

    if save_dir:
        out = save_dir / "knowledge_heatmap.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()

    plt.close()


def main():
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Run: uv run --group eval 07_evaluation/plot_tradeoffs.py")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="DIR", help="Save figures to directory instead of showing")
    parser.add_argument("--results-dir", default="data/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    save_dir = Path(args.save) if args.save else None

    results = {}
    for _, filename, _, _ in APPROACHES:
        r = load_result(results_dir / filename)
        if r is None:
            print(f"  Missing: {filename} (skipped)")
        results[filename] = r

    loaded = sum(1 for v in results.values() if v is not None)
    print(f"Loaded {loaded}/{len(APPROACHES)} result files")

    if loaded == 0:
        print("No results to plot. Run approach scripts first.")
        sys.exit(1)

    plot_tradeoff(results, save_dir)
    plot_knowledge_heatmap(results, save_dir)


if __name__ == "__main__":
    main()
