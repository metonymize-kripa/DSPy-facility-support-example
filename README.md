# Facility Support Classifier: Design Space Exploration

A systematic comparison of the full ML complexity ladder on a single narrow NLP task —
facility support message classification.

**Thesis**: LLMs are often over-engineered for domain-constrained classification tasks.
This project quantifies where simpler approaches (rules, FastText, classical ML) are
good enough, and where LLM capabilities genuinely earn their cost.

---

## The Task

Given an email to a facility support organisation, predict:
- **Urgency**: low / medium / high
- **Sentiment**: positive / neutral / negative
- **Categories**: one or more of 10 service-request labels (multi-label)

Dataset: [Meta llama-prompt-ops facility-support-analyzer](https://github.com/meta-llama/llama-prompt-ops/tree/main/use-cases/facility-support-analyzer) — 200 labeled emails.

References:
- Databricks and UC Berkeley guys: https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/
- Meta-llama guys: https://github.com/meta-llama/prompt-ops

Canonical metric: mean(urgency exact-match, sentiment exact-match, categories 10-way binary accuracy).
Published DSPy baseline (GPT-4.1-nano, zero-shot): **75.4%**.

---

## The LLM Axis: Gemma4 via Ollama

The LLM experiments are built around Google's **Gemma4** model family, run locally via Ollama.
No external API keys are required for the core LLM experiments.

| Model | Role in experiment |
|-------|--------------------|
| `gemma4:e4b` | **Student** — small, fast, cheap to run |
| `gemma4:26b` | **Teacher/Parent** — larger, used as reflection model in GEPA and as the upper reference |

The primary LLM experiment is a three-way comparison:

| Configuration | Model | Prompt |
|---------------|-------|--------|
| `parent/base` | gemma4:26b | zero-shot |
| `student/base` | gemma4:e4b | zero-shot |
| `student/compiled` | gemma4:e4b | GEPA-optimised |

Central question: **does GEPA prompt optimisation on gemma4:e4b close the gap to gemma4:26b zero-shot?**

---

## Project Structure

```
.
├── shared/                   # Shared dataset loader and canonical metrics
│   ├── dataset.py
│   └── metrics.py
│
├── 00_data_exploration/      # EDA: distributions, co-occurrence, vocabulary
├── 01_handwritten_rules/     # Regex/keyword cascade — no ML, no labels
├── 02_fasttext/              # FastText bag-of-n-grams
├── 03_classical_ml/          # TF-IDF + LogReg / SVM / Naive Bayes
├── 04_spacy/                 # spaCy textcat + entity-augmented hybrid
├── 05_cobweb_dm/             # Cobweb conceptual clustering (unsupervised)
├── 06_llm/                   # Gemma4 via Ollama, DSPy/GEPA optimisation
├── 07_evaluation/            # Cross-approach comparison, plots, failure analysis
│
├── data/                     # Cached dataset, results JSONs (gitignored)
├── archive/                  # Original prototype scripts (reference only)
│
├── pyproject.toml            # uv project root — all dependency groups
├── CONCEPTUAL_FRAMEWORK.md   # First-principles problem analysis
└── .env.example              # Optional API key template
```

---

## Quick Start

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Data exploration
uv run --group data 00_data_exploration/explore.py

# Handwritten rules baseline (no install needed beyond core deps)
uv run 01_handwritten_rules/classify.py

# FastText
uv run --group fasttext 02_fasttext/train.py

# Classical ML
uv run --group classical 03_classical_ml/train_eval.py

# LLM: Gemma4 base comparison (Ollama must be running)
ollama pull gemma4:e4b && ollama pull gemma4:26b
uv run --group llm 06_llm/compare_base.py

# LLM: GEPA optimisation
uv run --group llm 06_llm/gepa_optimise.py \
  --train-size 20 --val-size 10 --gepa-max-metric-calls 30

# Cross-approach comparison
uv run --group eval 07_evaluation/compare_all.py
```

---

## Dependency Groups

| Group | What it covers |
|-------|---------------|
| `data` | EDA (pandas, matplotlib, seaborn) |
| `rules` | Handwritten rules — stdlib only |
| `fasttext` | FastText training + inference |
| `classical` | scikit-learn classifiers |
| `spacy` | spaCy text classification pipelines |
| `cobweb` | Cobweb conceptual clustering |
| `llm` | DSPy + GEPA + Ollama/Gemma4 |
| `eval` | Cross-approach evaluation and plotting |

---

## Authors

Kripa & TBC(Wlodek)
