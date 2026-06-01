"""
07_evaluation/compare_all.py

Aggregate results from all approaches and print a comparison table.

Reads all data/results/*.json files and renders:
  1. Summary table: approach × {urgency, sentiment, categories, aggregate}
  2. Ranked table by aggregate score
  3. Paper-ready LaTeX table (optional, --latex flag)

Run:
    uv run 07_evaluation/compare_all.py
    uv run 07_evaluation/compare_all.py --latex
"""

import sys
import json
import argparse
from pathlib import Path

# Result files, in intended display order
RESULT_FILES = [
    ("Handwritten rules",        "01_handwritten_rules.json"),
    ("FastText",                 "02_fasttext.json"),
    ("Classical ML (best)",      "03_classical_ml_best.json"),
    ("spaCy entity-augmented",   "04_spacy_entity_augmented.json"),
    ("spaCy textcat",            "04_spacy_textcat.json"),
    ("gemma4:26b zero-shot",     "06_llm_parent_base.json"),
    ("gemma4:e4b zero-shot",     "06_llm_student_base.json"),
    ("gemma4:e4b + GEPA",        "06_llm_student_compiled.json"),
]


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return {
            "urgency":    data.get("urgency",    None),
            "sentiment":  data.get("sentiment",  None),
            "categories": data.get("categories", None),
            "aggregate":  data.get("aggregate",  None),
        }
    except Exception as e:
        print(f"  Warning: could not parse {path.name}: {e}")
        return None


def fmt(v) -> str:
    if v is None:
        return "    —   "
    return f" {v:.4f}"


def print_table(rows: list[tuple[str, dict | None]]):
    header = f"{'Approach':<35}  {'Urgency':>8}  {'Sentiment':>9}  {'Categories':>10}  {'Aggregate':>9}"
    print(header)
    print("-" * len(header))
    for label, r in rows:
        if r is None:
            print(f"  {label:<33}  {'(no results yet)':>40}")
            continue
        print(
            f"  {label:<33}"
            f"  {fmt(r['urgency']):>9}"
            f"  {fmt(r['sentiment']):>10}"
            f"  {fmt(r['categories']):>11}"
            f"  {fmt(r['aggregate']):>10}"
        )


def print_latex(rows: list[tuple[str, dict | None]]):
    print("\n% LaTeX table (paste into paper)")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Approach & Urgency & Sentiment & Categories & Aggregate \\")
    print(r"\midrule")
    for label, r in rows:
        if r is None:
            print(f"{label} & — & — & — & — \\\\")
            continue
        def lf(v):
            return "—" if v is None else f"{v:.3f}"
        print(f"{label} & {lf(r['urgency'])} & {lf(r['sentiment'])} & {lf(r['categories'])} & {lf(r['aggregate'])} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Comparison of approaches on the facility support classification task. "
          r"Urgency and sentiment are exact-match accuracy. "
          r"Categories is 10-way binary accuracy. "
          r"Aggregate is the arithmetic mean of the three.}")
    print(r"\label{tab:results}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX table")
    parser.add_argument("--results-dir", default="data/results", help="Path to results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run individual approach scripts first.")
        sys.exit(1)

    rows = []
    for label, filename in RESULT_FILES:
        r = load_result(results_dir / filename)
        rows.append((label, r))

    # Also check for any result files not in the predefined list
    known = {f for _, f in RESULT_FILES}
    extra = sorted(results_dir.glob("*.json"))
    for p in extra:
        if p.name not in known and not p.name.startswith("0") is False:
            pass  # skip non-approach files

    print("\n" + "="*70)
    print("FACILITY SUPPORT CLASSIFICATION: APPROACH COMPARISON")
    print("="*70)
    print_table(rows)

    # Ranked by aggregate
    ranked = [(l, r) for l, r in rows if r is not None and r["aggregate"] is not None]
    ranked.sort(key=lambda x: x[1]["aggregate"], reverse=True)

    if ranked:
        print(f"\n{'='*70}")
        print("RANKED BY AGGREGATE SCORE")
        print(f"{'='*70}")
        for i, (label, r) in enumerate(ranked, 1):
            print(f"  {i}. {label:<35} {r['aggregate']:.4f}")

        best = ranked[0]
        rules_row = next((r for l, r in rows if "rules" in l.lower() and r), None)
        if rules_row:
            gain = best[1]["aggregate"] - rules_row["aggregate"]
            print(f"\n  Best vs. handwritten rules: +{gain:.4f} ({gain*100:.1f} pp)")

    if args.latex:
        print_latex(rows)

    # Summary: missing results
    missing = [l for l, r in rows if r is None]
    if missing:
        print(f"\nMissing results ({len(missing)} approaches not yet run):")
        for l in missing:
            print(f"  - {l}")
    else:
        print(f"\nAll {len(rows)} approaches have results.")


if __name__ == "__main__":
    main()
