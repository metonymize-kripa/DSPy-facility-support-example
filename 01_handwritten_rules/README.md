# 01 — Handwritten Rules

A deterministic, regex/keyword classifier written without any ML.

## Motivation

Before fitting any model, establish what a domain expert with 30 minutes
and a text editor can achieve. This sets the floor and exposes which
sub-tasks are genuinely hard vs. solvable by pattern matching.

From the conceptual framework: a human expert uses a cheap-first decision
cascade — subject line scan → explicit urgency markers → keyword taxonomy
lookup → message arc reading. This script operationalises that cascade.

## Design

- Subject line patterns (fast, high-precision)
- Urgency keyword lists with escalation rules
- Category keyword vocabularies derived from label definitions
- Sentiment: negation-aware keyword matching + professional register heuristics
- Configurable: all rule lists are in `rules.py`, separate from the runner

## Scripts

| Script | Purpose |
|--------|---------|
| `rules.py` | All keyword lists, patterns, and decision logic |
| `classify.py` | Applies rules to dataset, reports per-task accuracy |

## Run

```bash
uv run 01_handwritten_rules/classify.py
```

## Expected outcome

~65–75% aggregate. Strong on explicit urgency and clear-category messages.
Weak on polite complaints (sentiment) and intent-based category disambiguation.
Failure mode analysis from this script directly informs what the ML approaches need to fix.
