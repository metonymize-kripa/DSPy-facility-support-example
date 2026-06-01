# 02 — FastText

Facebook's bag-of-n-grams supervised classifier. Trains in seconds on CPU.
Represents the "cheapest learned model" point on the design space ladder.

## Motivation

FastText is the canonical answer to "what's the minimum viable ML solution
for text classification?" It was the state of the art for customer support
routing in 2016–2018. The question is how much it degrades on a 200-example
dataset vs. the LLM baselines.

## Design choices

The task has three outputs with different structures:

| Sub-task | FastText formulation |
|----------|---------------------|
| Urgency | 3-class single-label |
| Sentiment | 3-class single-label |
| Categories | 10-label multi-label (one model per label, or single multi-label model) |

Both single multi-label and per-label binary approaches will be tried.

FastText requires converting the dataset to its training format:
`__label__<label> <text>` — handled by `prepare_data.py`.

## Scripts

| Script | Purpose |
|--------|---------|
| `prepare_data.py` | Convert dataset to FastText training format |
| `train.py` | Train and cross-validate FastText models |
| `classify.py` | Evaluate on held-out test set using shared metrics |

## Run

```bash
uv run --group fasttext 02_fasttext/train.py
uv run --group fasttext 02_fasttext/classify.py
```

## Known limitations

- 200 examples is very small for FastText (designed for millions)
- n-gram features won't capture negation or professional register masking
- No world knowledge for implicit urgency

Expected outcome: ~65–72% aggregate, strong on categories (vocabulary signal),
weak on sentiment (register masking) and implicit urgency.
