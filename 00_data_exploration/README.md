# 00 — Data Exploration

Characterise the dataset before fitting any model.

## Goals
- Label distribution: urgency, sentiment, category frequencies
- Category co-occurrence matrix (which labels appear together?)
- Message length distribution (characters, tokens)
- Vocabulary analysis: top n-grams per category
- Inter-label correlation: does high urgency predict certain categories?
- Identify hard cases: ambiguous labels, short/atypical messages

## Scripts

| Script | Purpose |
|--------|---------|
| `explore.py` | Full EDA — distributions, co-occurrence, n-gram analysis |
| `label_stats.py` | Quick label counts and co-occurrence table |

## Run

```bash
uv run --group data 00_data_exploration/explore.py
```
