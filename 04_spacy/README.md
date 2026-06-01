# 04 — spaCy

spaCy's `textcat_multilabel` pipeline for categories, plus custom
rule-based components for urgency and sentiment.

## Motivation

spaCy sits between classical ML and neural models. It supports:
- Supervised multi-label text classification (`textcat_multilabel`)
- Rule-based matching via `Matcher` and `PhraseMatcher`
- Named entity recognition — extractable as structured features

The neuro-symbolic angle from the conceptual framework lives here:
entity extraction (equipment type, location, temporal expressions)
as structured features fed into a downstream classifier.

## Approach

1. **Base spaCy textcat**: train `en_core_web_sm` + `textcat_multilabel` component on the dataset
2. **Rule + textcat hybrid**: hand-crafted urgency/sentiment rules + trained category classifier
3. **Entity-augmented**: extract entities (HVAC, mold, carpet, leak, etc.) via PhraseMatcher,
   encode as binary features, concatenate with TF-IDF → LogReg (bridging 03 and 04)

## Scripts

| Script | Purpose |
|--------|---------|
| `train_textcat.py` | Train spaCy textcat_multilabel pipeline |
| `entity_rules.py` | PhraseMatcher entity vocabulary for facility domain |
| `hybrid_classify.py` | Rule-based urgency/sentiment + textcat categories |
| `entity_augmented.py` | Entity features + classical ML |

## Run

```bash
# Download spaCy model first (one-time)
uv run --group spacy python -m spacy download en_core_web_sm

uv run --group spacy 04_spacy/train_textcat.py
uv run --group spacy 04_spacy/hybrid_classify.py
```

## Expected outcome

Similar aggregate to classical ML. The entity-augmented variant tests whether
structured extraction closes the urgency gap (domain knowledge via entity vocab)
without needing a large LLM.
