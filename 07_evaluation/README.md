# 07 — Evaluation

Cross-approach comparison harness. Aggregates results from all approaches
and produces the tables and charts for the paper.

## Design

Each approach script writes its results to `data/results/<approach>.json`
in a canonical schema. This folder reads all results and produces:

- Aggregate score table (all approaches, all sub-tasks)
- Accuracy vs. cost-per-query scatter plot
- Accuracy vs. training-data-required plot
- Per-sub-task breakdown: which approach wins on urgency vs. sentiment vs. categories?
- Failure mode analysis: where do rules fail that LLMs fix? Where do LLMs fail that rules don't?

## Results schema (`data/results/<approach>.json`)

```json
{
  "approach": "fasttext",
  "variant": "multilabel",
  "model": null,
  "n_test": 68,
  "aggregate": 0.71,
  "urgency": 0.74,
  "sentiment": 0.63,
  "categories": 0.76,
  "cost_per_query_usd": 0.0,
  "latency_p50_ms": 2,
  "latency_p95_ms": 5,
  "training_examples_required": 132,
  "notes": "multilabel fasttext, wordNgrams=2"
}
```

## Scripts

| Script | Purpose |
|--------|---------|
| `compare_all.py` | Load all result JSONs, print comparison table |
| `plot_tradeoffs.py` | Accuracy vs. cost scatter, per-task breakdown charts |
| `failure_analysis.py` | Find examples where approaches disagree; characterise errors |

## Run

```bash
uv run --group eval 07_evaluation/compare_all.py
uv run --group eval 07_evaluation/plot_tradeoffs.py
```
