# Implementation Plan

Systematic execution guide for the facility support design-space study.
Each phase produces runnable code, a results JSON, and a FINDINGS.md entry.
Phases are ordered so each one informs the next.

---

## How findings bubble up

Every approach script, when it finishes, does two things:

1. **Writes** `data/results/<approach>.json` — machine-readable scores in the canonical schema
2. **Prints** a structured "findings block" to stdout that the researcher copies into `FINDINGS.md`

The `07_evaluation/compare_all.py` script reads all result JSONs and produces the
master comparison table and plots. `FINDINGS.md` is the human-readable record
of observations, surprises, and interpretations accumulated as experiments run.

---

## Results JSON schema (canonical, used by all approaches)

```json
{
  "approach":                 "fasttext",
  "variant":                  "multilabel",
  "model":                    null,
  "n_test":                   68,
  "aggregate":                0.71,
  "urgency":                  0.74,
  "sentiment":                0.63,
  "categories":               0.76,
  "cost_per_query_usd":       0.0,
  "latency_p50_ms":           2,
  "latency_p95_ms":           5,
  "training_examples_required": 132,
  "notes":                    "wordNgrams=2, minCount=1"
}
```

---

## Phase 0 — Foundation (DONE)

**Status**: Complete.

Deliverables already in place:
- `CONCEPTUAL_FRAMEWORK.md` — first-principles problem analysis
- `shared/dataset.py` — canonical data loader
- `shared/metrics.py` — canonical metric (10-way binary accuracy, matching DSPy tutorial)
- Project structure, `pyproject.toml`, dependency groups

**Before proceeding**: verify `shared/dataset.py` loads correctly.
```bash
uv run python -c "from shared.dataset import load_examples; ex = load_examples(); print(len(ex), ex[0].keys())"
```
Expected: `200 dict_keys(['message', 'urgency', 'sentiment', 'categories'])`

---

## Phase 1 — Data Exploration

**Goal**: Characterise the dataset before fitting anything. Findings here
constrain what approaches are feasible and what failure modes to expect.

**Script to build**: `00_data_exploration/explore.py`

### What to implement

1. **Label distributions**
   - Urgency: count of low/medium/high
   - Sentiment: count of positive/neutral/negative
   - Per-category frequency (how often is each label True?)

2. **Category co-occurrence matrix**
   - 10×10 matrix: for each pair of categories, how often do they appear together?
   - Reveals which labels are almost never multi-assigned (near-exclusive)
   - and which are frequent co-labels (e.g., `general_inquiries` + anything)

3. **Message length distribution**
   - Character count histogram
   - Word count histogram
   - Does length correlate with urgency or category?

4. **Vocabulary per label**
   - Top 20 unigrams and bigrams per active category (TF-IDF weighted)
   - For urgency levels: top words for "high" vs "low"
   - This tells us whether FastText's lexical signal will be sufficient

5. **Hard case identification**
   - Messages where sentiment ≠ intuitive (positive-register complaints)
   - Messages with 3+ active categories
   - Very short messages (< 50 words) — harder for all classifiers

### Run
```bash
uv run --group data 00_data_exploration/explore.py
```

### Produces
- Console output with all statistics
- `data/results/00_data_exploration.json` with key counts
- Matplotlib figures saved to `00_data_exploration/figures/`

### Key questions to answer in FINDINGS.md
- Is `general_inquiries` the dominant catch-all? (expected: yes)
- Which category is rarest? (determines class imbalance risk)
- Do urgency=high messages use distinct vocabulary? (determines FastText ceiling)
- How many messages have register-masked sentiment? (determines LLM advantage)

---

## Phase 2 — Handwritten Rules Baseline

**Goal**: Establish what a domain expert with regex and keywords achieves,
with no ML and no labeled data. This is the true floor — any learned model
that can't beat this needs to be reconsidered.

**Scripts to build**:
- `01_handwritten_rules/rules.py` — all rule definitions, no I/O
- `01_handwritten_rules/classify.py` — applies rules, reports results

### What to implement

**Urgency rules** (cascade — first match wins):
1. Subject line contains urgency keywords → high
2. Body contains safety/hazard language → high
3. Body contains explicit time pressure ("today", "tonight", "immediately") → high
4. Body contains "soon", "when convenient", "at your earliest" → medium
5. Body is an inquiry with no problem report → low
6. Default → medium

**Category rules** (non-exclusive — all matching rules fire):
- Keyword vocabularies for each of the 10 categories
- Derived directly from the label definitions in CONCEPTUAL_FRAMEWORK.md
- Subject-line priority: subject match overrides body match for conflicting signals

**Sentiment rules**:
1. Explicit complaint language ("upset", "disappointed", "unacceptable") → negative
2. Explicit praise ("excellent", "wonderful", "very satisfied") → positive
3. Register trap heuristic: positive opener + complaint body → neutral
4. Default → neutral

### Run
```bash
uv run 01_handwritten_rules/classify.py
```

### Produces
- `data/results/01_handwritten_rules.json`
- Per-rule firing frequency (which rules triggered most?)
- False positive / false negative breakdown by category

### Key questions for FINDINGS.md
- What aggregate score do pure rules achieve?
- Which sub-task do rules perform worst on? (expected: sentiment, due to register)
- Which categories are rule-addressable vs. require learning?
- Where exactly do rules fail? (seed the failure analysis in Phase 7)

---

## Phase 3 — FastText

**Goal**: Minimum viable supervised ML. Bag-of-n-grams, trains in seconds.

**Scripts to build**:
- `02_fasttext/prepare_data.py` — convert dataset to FastText format
- `02_fasttext/train.py` — train models with cross-validation
- `02_fasttext/classify.py` — evaluate on held-out test set

### What to implement

Three separate FastText models:
1. **Urgency**: single-label 3-class (`__label__low`, `__label__medium`, `__label__high`)
2. **Sentiment**: single-label 3-class
3. **Categories**: multi-label (FastText supports `__label__cat1 __label__cat2 ...` natively)

Key hyperparameters to try (via cross-validation on train set):
- `wordNgrams`: 1 vs 2 vs 3
- `epoch`: 25, 50, 100
- `lr`: 0.1, 0.5, 1.0
- `minCount`: 1 (small dataset — keep all tokens)

**Data note**: 200 examples is small for FastText. Use the full train+val set
for training and report on test. Cross-validate only within train+val.

### Run
```bash
uv run --group fasttext 02_fasttext/prepare_data.py
uv run --group fasttext 02_fasttext/train.py
uv run --group fasttext 02_fasttext/classify.py
```

### Produces
- `data/results/02_fasttext.json`
- Trained model files in `02_fasttext/models/` (gitignored)

### Key questions for FINDINGS.md
- How close does FastText get to the DSPy GPT-4.1-nano baseline (75.4%)?
- Which sub-task does it struggle with most?
- Does bigram vs. unigram matter on 200 examples?

---

## Phase 4 — Classical ML

**Goal**: TF-IDF + scikit-learn classifiers. Often outperforms FastText on
small datasets due to better regularisation. Also provides feature importance
for interpretability.

**Script to build**: `03_classical_ml/train_eval.py`

### What to implement

For urgency and sentiment: `sklearn.pipeline.Pipeline` with:
- `TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)`
- Classifiers: LogisticRegression, LinearSVC, MultinomialNB

For categories: wrap each in `MultiOutputClassifier`.

Report per-classifier, per-task accuracy. Use 5-fold CV on train+val,
final evaluation on held-out test.

**Bonus**: `03_classical_ml/feature_analysis.py`
- Top 20 TF-IDF features per label from LogReg coefficients
- Validates (or contradicts) the keyword intuitions from the rules approach

### Run
```bash
uv run --group classical 03_classical_ml/train_eval.py
uv run --group classical 03_classical_ml/feature_analysis.py
```

### Produces
- `data/results/03_classical_ml_logreg.json`
- `data/results/03_classical_ml_svm.json`
- `data/results/03_classical_ml_nb.json`
- Feature weight table printed to console

### Key questions for FINDINGS.md
- Does LogReg outperform FastText? By how much?
- Do the top features per category match the handwritten rule vocabularies?
- Which classifier is most robust on this dataset size?

---

## Phase 5 — spaCy

**Goal**: spaCy's `textcat_multilabel` pipeline + entity-augmented hybrid.
Tests the neuro-symbolic angle: does extracting structured entities
(equipment type, urgency markers) help a downstream classifier?

**Scripts to build**:
- `04_spacy/train_textcat.py`
- `04_spacy/entity_rules.py` — PhraseMatcher vocabulary
- `04_spacy/entity_augmented.py` — entity features + LogReg

### What to implement

**5a — textcat_multilabel**:
Standard spaCy training pipeline for all three outputs.
Compare against classical ML.

**5b — Entity-augmented** (the interesting variant):
1. Define a domain entity vocabulary via PhraseMatcher:
   - Equipment: HVAC, carpet, plumbing, mold, elevator, window, ...
   - Urgency markers: immediately, ASAP, emergency, today, ...
   - Sentiment markers: upset, disappointed, satisfied, appreciate, ...
   - Service type: cleaning, maintenance, repair, inspection, ...
2. Run PhraseMatcher over each message → binary feature vector
3. Concatenate with TF-IDF → LogReg
4. Compare against plain TF-IDF LogReg from Phase 4

**Hypothesis**: entity features should improve urgency accuracy specifically,
because they encode domain knowledge (HVAC + urgency marker = emergency)
that TF-IDF misses.

### Run
```bash
uv run --group spacy python -m spacy download en_core_web_sm
uv run --group spacy 04_spacy/train_textcat.py
uv run --group spacy 04_spacy/entity_augmented.py
```

### Produces
- `data/results/04_spacy_textcat.json`
- `data/results/04_spacy_entity_augmented.json`

### Key questions for FINDINGS.md
- Does entity augmentation improve urgency accuracy over plain TF-IDF?
- If yes: this is evidence that domain knowledge (Type 2 from the framework)
  can be injected cheaply without an LLM
- How much of the LLM advantage on urgency is just this?

---

## Phase 6 — Cobweb / Conceptual Clustering

**Goal**: Unsupervised discovery. Does the data have natural cluster structure
that aligns with the task labels — without any training signal?

**Scripts to build**:
- `05_cobweb_dm/cobweb_keywords.py` — interpretable feature space
- `05_cobweb_dm/cobweb_text.py` — TF-IDF feature space

### What to implement

**6a — Keyword feature Cobweb** (do this first):
- Binary features: is each of ~50 domain keywords present?
- Run Cobweb incrementally on the 200 examples
- For each discovered concept, compute:
  - Purity: what fraction of examples in this concept share the same urgency/category?
  - Label entropy: how mixed is the concept?
- Print the concept tree

**6b — TF-IDF Cobweb** (secondary):
- Discretise TF-IDF into binary (top-k terms present/absent)
- Same evaluation

**Key question**: Does Cobweb without labels recover urgency levels or
category clusters? If purity is high, the task has strong natural structure.
If purity is low, the task requires labeled supervision — unsupervised methods won't work.

**Dataset size**: 200 examples is well within Cobweb's intended range.
This is one of the few approaches where small N is not a disadvantage.

### Run
```bash
uv run --group cobweb 05_cobweb_dm/cobweb_keywords.py
uv run --group cobweb 05_cobweb_dm/cobweb_text.py
```

### Produces
- `data/results/05_cobweb_keywords.json` (cluster purity scores)
- `data/results/05_cobweb_text.json`
- Concept tree printed to console

### Key questions for FINDINGS.md
- What is the maximum cluster purity for urgency? For categories?
- Which categories are most "natural" (high purity without labels)?
- Does keyword Cobweb outperform TF-IDF Cobweb (interpretability vs. coverage)?

---

## Phase 7 — LLM: Gemma4 via Ollama

**Goal**: Run the three-way comparison that is the core LLM experiment.

**Scripts to build**:
- `06_llm/compare_base.py` — parent/base vs student/base, no GEPA (fast)
- `06_llm/gepa_optimise.py` — full GEPA loop + three-way comparison
- `06_llm/thinking_ablation.py` — think=on vs think=off for gemma4:e4b

### Step 7a — Base comparison first (no GEPA)

Run both models zero-shot on the full test set.
This gives the unoptimised gap that GEPA needs to close.

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

uv run --group llm 06_llm/compare_base.py \
  --student-model ollama/gemma4:e4b \
  --parent-model ollama/gemma4:26b \
  --test-size 68
```

Produces:
- `data/results/06_llm_student_base.json`
- `data/results/06_llm_parent_base.json`

### Step 7b — GEPA optimisation

```bash
uv run --group llm 06_llm/gepa_optimise.py \
  --student-model ollama/gemma4:e4b \
  --reflection-model ollama/gemma4:26b \
  --train-size 20 \
  --val-size 10 \
  --gepa-max-metric-calls 30 \
  --comparison-size 68
```

Start with `--gepa-max-metric-calls 15` for a quick smoke test,
then `30` for paper-quality results, `60` for full budget.

Produces:
- `data/results/06_llm_student_compiled.json`
- Optimised prompts printed to console (copy into `FINDINGS.md`)

### Step 7c — Thinking ablation

```bash
uv run --group llm 06_llm/thinking_ablation.py \
  --model ollama/gemma4:e4b \
  --test-size 68
```

Produces:
- `data/results/06_llm_student_think_on.json`
- `data/results/06_llm_student_think_off.json`

### Key questions for FINDINGS.md
- What is the unoptimised gap: student/base vs parent/base?
- Does GEPA close it? Fully, partially, or not at all?
- Which sub-task benefits most from GEPA? (expected: sentiment and urgency)
- Does thinking mode matter for the student model?
- What do the GEPA-evolved prompts look like? (qualitative analysis)

---

## Phase 8 — Cross-approach Evaluation

**Goal**: Assemble all results, produce paper-ready tables and plots,
run failure analysis.

**Scripts to build**:
- `07_evaluation/compare_all.py`
- `07_evaluation/plot_tradeoffs.py`
- `07_evaluation/failure_analysis.py`

### Step 8a — Master comparison table

Reads all `data/results/*.json`, produces:

```
Approach                    Aggregate  Urgency  Sentiment  Categories  Cost/query  Train N
─────────────────────────────────────────────────────────────────────────────────────────
01 Handwritten rules           ??.?%    ??.?%     ??.?%      ??.?%       $0.000       0
02 FastText                    ??.?%    ??.?%     ??.?%      ??.?%       $0.000     132
03 Classical ML (LogReg)       ??.?%    ??.?%     ??.?%      ??.?%       $0.000     132
04 spaCy textcat               ??.?%    ??.?%     ??.?%      ??.?%       $0.000     132
04 spaCy entity-augmented      ??.?%    ??.?%     ??.?%      ??.?%       $0.000     132
05 Cobweb (keywords)           ??.?%    ??.?%     ??.?%      ??.?%       $0.000       0
06 Gemma4:e4b zero-shot        ??.?%    ??.?%     ??.?%      ??.?%       $0.000       0
06 Gemma4:26b zero-shot        ??.?%    ??.?%     ??.?%      ??.?%       $0.000       0
06 Gemma4:e4b + GEPA           ??.?%    ??.?%     ??.?%      ??.?%       $0.000      20
── DSPy tutorial (GPT-4.1-nano zero-shot, published baseline) ─── 75.4% ──────────────
```

### Step 8b — Trade-off plots

- Scatter: aggregate accuracy vs. training examples required
- Scatter: aggregate accuracy vs. latency (log scale)
- Bar chart: per-sub-task accuracy by approach
- Line chart: GEPA score progression across optimisation iterations

### Step 8c — Failure analysis

Find examples where approaches diverge:
- Rules correct, LLM wrong (what lexical patterns do rules catch that LLMs miss?)
- LLM correct, rules wrong (what requires reasoning beyond keywords?)
- All approaches wrong (what are the genuinely hard cases?)

Sample 10 examples from each failure category for qualitative analysis.
These go directly into the paper's error analysis section.

### Run
```bash
uv run --group eval 07_evaluation/compare_all.py
uv run --group eval 07_evaluation/plot_tradeoffs.py
uv run --group eval 07_evaluation/failure_analysis.py
```

---

## Phase 9 — Paper Write-up

**Skeleton**: `paper/` folder with section stubs.
Each section maps directly to findings accumulated in `FINDINGS.md`.

| Paper section | Populated from |
|--------------|---------------|
| 1. Introduction | Conceptual framework §1, §7 |
| 2. Task and dataset | Phase 1 findings (data exploration) |
| 3. Problem decomposition | Conceptual framework §3, §5 |
| 4. Approaches | Per-phase READMEs |
| 5. Experiments | Phase 8 master table + plots |
| 6. Analysis | Phase 8 failure analysis + per-sub-task breakdown |
| 7. Discussion | FINDINGS.md observations |
| 8. Conclusion | Conceptual framework §7 + empirical confirmation |

---

## Execution order and dependencies

```
Phase 0 (done)
    └── Phase 1 (data exploration) ← run first, no deps
            ├── Phase 2 (rules)     ← can start immediately after Phase 1
            ├── Phase 3 (FastText)  ← can run in parallel with Phase 2
            ├── Phase 4 (classical) ← can run in parallel
            ├── Phase 5 (spaCy)     ← can run in parallel
            ├── Phase 6 (Cobweb)    ← can run in parallel
            └── Phase 7 (LLM)       ← can run in parallel; Ollama must be running
                    └── Phase 8 (evaluation) ← needs all results JSONs
                            └── Phase 9 (paper)
```

Phases 2–7 are independent of each other and can run in any order or in parallel.
The only hard dependency is Phase 8 requires all result JSONs to be present.

---

## Time estimates

| Phase | Estimated time to implement | Estimated time to run |
|-------|----------------------------|----------------------|
| 1 Data exploration | 1–2 hours | < 1 min |
| 2 Handwritten rules | 2–3 hours | < 1 min |
| 3 FastText | 1–2 hours | < 5 min |
| 4 Classical ML | 1–2 hours | < 5 min |
| 5 spaCy | 2–3 hours | 5–15 min |
| 6 Cobweb | 2–3 hours | < 5 min |
| 7 LLM base comparison | 1 hour | 10–30 min (Ollama inference) |
| 7 LLM GEPA optimisation | 1 hour | 30 min – 2 hours (depends on budget) |
| 8 Evaluation | 2–3 hours | < 5 min |
| 9 Paper | ongoing | — |

**Total implementation**: ~15–20 hours of coding
**Total run time**: 2–4 hours (dominated by GEPA optimisation)
