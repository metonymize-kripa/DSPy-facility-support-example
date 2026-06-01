# 06 — LLM Approaches (Gemma4 via Ollama + DSPy/GEPA)

LLM-based classifiers using Google's Gemma4 model family, run locally via Ollama.
DSPy provides the program structure; GEPA provides prompt optimisation.

All models run locally — no API keys required for the core LLM experiments.

## The Core Comparison

The primary experiment replicates and extends `archive/dspy_support_analyzer.py`:

| Run | Model | Prompt | Role |
|-----|-------|--------|------|
| `parent/base` | `gemma4:26b` | unoptimised | Upper reference: what does the big model do zero-shot? |
| `student/base` | `gemma4:e4b` | unoptimised | Lower reference: what does the small model do zero-shot? |
| `student/compiled` | `gemma4:e4b` | GEPA-optimised | Key result: does optimisation close the gap to the big model? |

This is the concrete instantiation of the paper's central question:
**can prompt optimisation on a 4B model substitute for running a 26B model?**

## Thinking mode

Gemma4 supports a `think=True/False` parameter in Ollama.
The archive scripts default to `think=False` for the student (speed)
and `unset` for the reflection model (let the model decide).
Both modes are worth benchmarking — thinking adds latency but may improve accuracy.

## Scripts

| Script | Purpose |
|--------|---------|
| `compare_base.py` | Run parent/base vs student/base on the test set — no GEPA |
| `gepa_optimise.py` | Run full GEPA loop, then evaluate all three configurations |
| `thinking_ablation.py` | Compare think=on vs think=off for gemma4:e4b |

## Run

```bash
# Prerequisite: Ollama running with both models pulled
ollama pull gemma4:e4b
ollama pull gemma4:26b

# Optional: apply Ollama performance settings for concurrent DSPy eval
bash archive/reset_ollama_for_dspy.sh

# Base comparison (no GEPA) — fast, good starting point
uv run --group llm 06_llm/compare_base.py

# Full GEPA optimisation
uv run --group llm 06_llm/gepa_optimise.py \
  --student-model ollama/gemma4:e4b \
  --reflection-model ollama/gemma4:26b \
  --train-size 20 --val-size 10 \
  --gepa-max-metric-calls 30

# Thinking ablation
uv run --group llm 06_llm/thinking_ablation.py
```

## Key parameters (from archive/dspy_support_analyzer.py)

```
--student-model         ollama/gemma4:e4b    (default)
--reflection-model      ollama/gemma4:26b    (default)
--student-num-ctx       8192
--reflection-num-ctx    32768
--student-thinking      off      (think=False)
--reflection-thinking   unset    (model default)
--train-size            10       (quick default; use 20–40 for paper results)
--val-size              5        (quick default; use 10–20 for paper results)
--gepa-max-metric-calls 15       (quick default; use 30–60 for paper results)
--comparison-size       10
```

## Where results go

Each script writes to `data/results/llm_<variant>.json` using the shared results schema.
