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

### Key Results (this study)

| Approach | Aggregate | Latency |
|----------|-----------|---------|
| **SVM (TF-IDF)** | **86.7%** 🏆 | 2ms |
| FastText | 84.4% | 1ms |
| Entity-augmented spaCy | 81.9% | 10ms |
| qwen3.6:35b-mlx zero-shot | 78.7% | ~9.4s |
| gemma4:e4b + GEPA | 78.4% | ~9.4s |
| Handwritten rules | 66.6% | 1ms |

**Finding**: Classical ML (SVM) beats zero-shot LLMs and GEPA-optimized prompts on this narrow-domain task. See `FINDINGS.md` and `ENGINEERING_GUIDE.md` for detailed analysis.

---

## The LLM Axis: Local Models via Ollama

The LLM experiments use Google's **Gemma4** (4B) as the student model and **Qwen3.6** (35B) as the reflection/parent model, run locally via Ollama. No external API keys required.

| Model | Role in experiment |
|-------|--------------------|
| `gemma4:e4b` (4B) | **Student** — small, fast, cheap to run |
| `qwen3.6:35b-mlx` (35B) | **Parent/Reflection** — larger model for GEPA optimization |

The primary LLM experiment is a three-way comparison:

| Configuration | Model | Prompt |
|---------------|-------|--------|
| `parent/base` | qwen3.6:35b-mlx | zero-shot |
| `student/base` | gemma4:e4b | zero-shot |
| `student/compiled` | gemma4:e4b | GEPA-optimised |

Central question: **does GEPA prompt optimisation on gemma4:e4b close the gap to qwen3.6:35b-mlx zero-shot?**

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
├── 06_llm/                   # Gemma4 via Ollama, DSPy/GEPA optimisation
├── 07_evaluation/            # Cross-approach comparison, plots, failure analysis
│
├── data/                     # Cached dataset, results JSONs (gitignored)
├── archive/                  # Original prototype scripts (reference only)
│
├── pyproject.toml            # uv project root — all dependency groups
├── CONCEPTUAL_FRAMEWORK.md   # First-principles problem analysis
├── ENGINEERING_GUIDE.md      # Engineering trade-offs and recommendations
├── FINDINGS.md               # Detailed experimental results
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
ollama pull gemma4:e4b && ollama pull qwen3.6:35b-mlx
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
| `llm` | DSPy + GEPA + Ollama |
| `eval` | Cross-approach evaluation and plotting |

---

## Authors

Kripa & TBC(Wlodek)
