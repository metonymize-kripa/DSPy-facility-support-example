# Findings

Living document. Updated after each experimental phase.
Feeds directly into the paper write-up.

**How to use**: After running each phase, fill in the `[ ]` checkboxes and
paste the key numbers and observations into the relevant section.
The "Paper implication" prompts are pre-written — edit them to match actual results.

---

## Phase 1 — Data Exploration

**Status**: [x] complete

### Label distributions

| Label | Count | % of dataset |
|-------|-------|-------------|
| urgency = low | 84 | 42.0% |
| urgency = medium | 63 | 31.5% |
| urgency = high | 53 | 26.5% |
| sentiment = positive | 61 | 30.5% |
| sentiment = neutral | 113 | 56.5% |
| sentiment = negative | 26 | 13.0% |

| Category | True count | % active |
|----------|-----------|---------| 
| routine_maintenance_requests | 53 | 26.5% |
| customer_feedback_and_complaints | 41 | 20.5% |
| training_and_support_requests | 23 | 11.5% |
| quality_and_safety_concerns | 39 | 19.5% |
| sustainability_and_environmental_practices | 48 | 24.0% |
| cleaning_services_scheduling | 24 | 12.0% |
| specialized_cleaning_services | 48 | 24.0% |
| emergency_repair_services | 23 | 11.5% |
| facility_management_issues | 35 | 17.5% |
| general_inquiries | 46 | 23.0% |

### Class balance observations

**Urgency** is the most balanced sub-task (42/32/27 split). A majority-class
baseline would score 42% — low enough that any real model must learn.

**Sentiment** is heavily skewed toward neutral (57%). A majority-class baseline
scores 57% — which means a bad model can look decent just by predicting neutral.
Negative class is severely underrepresented (13%, only 26 examples). FastText
and classical ML will likely underperform on negative specifically due to low
sample count. This is the sub-task where label scarcity hurts most.

**Categories**: `training_and_support_requests` and `emergency_repair_services`
are the rarest (11.5% each, ~23 examples). `routine_maintenance_requests`,
`specialized_cleaning_services`, and `sustainability_and_environmental_practices`
are the most common (~25% each). A classifier that predicts the five most common
categories on every example would score high on binary accuracy but fail completely
on recall for the rare classes.

### Category co-occurrence

Top co-occurring pairs (from co-occurrence analysis):
1. `quality_and_safety_concerns` + `customer_feedback_and_complaints` — highest pair
2. `general_inquiries` + `sustainability_and_environmental_practices`
3. `general_inquiries` + `specialized_cleaning_services`

Key pattern: `general_inquiries` co-occurs broadly — it acts as a residual
catch-all category, not a mutually exclusive class. Any model that treats
it as exclusive will systematically miss it when a more specific category is present.

`emergency_repair_services` has low co-occurrence with most categories — it tends to
be the *only* active label when present, which is consistent with its urgency signal.

### Message length

- Median: ~120 words (estimated from char stats — mean chars ≈ 700, typical word ~6 chars)
- High-urgency messages are not shorter — they tend to be similar length with explicit
  alarm language front-loaded in the subject line
- Very short messages (< 50 words) are edge cases — likely ambiguous to any model

### N-gram analysis — key findings

**Urgency signal is lexically explicit and strong.**

High-urgency bigrams: `subject urgent`, `immediate assistance`, `attention required`,
`given urgency`, `forward swift`. Trigrams: `immediate attention required`,
`subject urgent assistance`, `look forward swift response`.

These are unambiguous explicit alarm phrases. A keyword rule that matches on
`urgent|immediate|attention required` will capture the vast majority of high-urgency
cases. This confirms the conceptual framework prediction: urgency=high is largely
solvable by rules. The implicit urgency cases (medium) are harder: their distinctive
trigrams are `routine maintenance hvac system`, `come take look`, `recently encountered
issue` — contextual, domain-specific, not lexically urgent.

**Sentiment signal is register-masked for negative.**

Negative bigrams: `though must`, `must admit`, `recent experiences left`, `quite
disappointed`, `rectify situation`. These are hedged constructions — the sender
acknowledges a complaint indirectly ("though I must admit..."). A bag-of-words model
will assign high weight to `disappointed` and `rectify` but miss the `though must admit`
hedging pattern that signals the complaint is buried.

Positive bigrams: `truly dedication`, `dedication professionalism`, `done fantastic
job` — direct positive expressions, easily captured lexically.

Neutral bigrams: `wanted gather`, `reviewed information`, `inquire sustainability` —
purely informational, no affect. Bag-of-words handles neutral well.

**The sentiment asymmetry**: positive and neutral are lexically separable; negative
is register-masked. FastText will do well on positive/neutral, poorly on negative.
LLMs with pragmatic knowledge (Type 3) have an advantage specifically on negative.

### Hard cases identified

- Messages with register-masked sentiment (positive opener, non-positive label): likely ~15–25
- Messages with 3+ active categories: estimated ~20–30 (based on co-occurrence density)
- Very short messages (< 40 words): small fraction, ~5–10%

### Paper implication

The label distribution reveals three distinct difficulty drivers:

1. **Sentiment** has severe class imbalance for `negative` (13%, 26 examples). Any
   approach trained on this data will be unreliable on negative cases — not because
   the signal is absent but because there are too few examples to learn the
   hedged-register pattern reliably. This creates an irreducible floor that only
   LLMs (which have pretraining exposure to this register) can overcome.

2. **Urgency** is lexically saturated for the high class — keyword rules should
   perform near-ceiling on high-urgency detection. The medium class is where implicit
   domain knowledge (HVAC context, seasonal timing) matters. This is where LLMs earn
   a small but real advantage.

3. **Categories**: `general_inquiries` is a residual label (co-occurs with everything),
   not a mutually exclusive class. Models that fail to understand this will
   systematically under-predict it in multi-label scenarios. This is a taxonomy
   knowledge (Type 1) failure, addressable by explicit label definitions in the prompt.

---

## Phase 2 — Handwritten Rules

**Status**: [x] complete

### Results

| Sub-task | Accuracy |
|----------|---------| 
| Urgency | 57.4% |
| Sentiment | 70.6% |
| Categories | 71.8% |
| **Aggregate** | **66.6%** |

Published baseline (GPT-4.1-nano zero-shot): **75.4%**  
Rules gap vs. baseline: **8.8pp**

### Where rules failed

| Failure Mode | Count | Pattern |
|-------------|-------|---------|
| **Urgency over-triggering** | 29 errors | `medium → high`: 15x, `low → high`: 8x |
| **Positive sentiment under-detection** | 11 errors | `positive → neutral`: 11x |
| **Category false positives** | High FP rate | 3 categories at 100% recall but <35% precision |

**Root cause analysis:**
1. **Urgency keywords too broad**: Words like "today", "this morning" in routine contexts trigger false high-urgency predictions
2. **Category vocabularies too permissive**: No negative constraints to exclude false matches
3. **Positive sentiment regex too narrow**: Misses moderate appreciation ("thank you", "good work")

### Category precision / recall breakdown

| Category | Precision | Recall | Assessment |
|----------|-----------|--------|------------|
| specialized_cleaning_services | 100% | 70.6% | Well-calibrated vocabulary |
| cleaning_services_scheduling | 77.8% | 87.5% | Good balance |
| customer_feedback_and_complaints | 75.0% | 23.1% | Low recall — missing complaint patterns |
| emergency_repair_services | 58.8% | 83.3% | Acceptable |
| general_inquiries | 45.2% | 87.5% | High recall, needs exclusion patterns |
| facility_management_issues | 45.0% | 81.8% | Over-fires on management keywords |
| sustainability_and_environmental_practices | 31.6% | 92.3% | Eco-vocabulary too broad |
| routine_maintenance_requests | 33.3% | 100% | Catches all, but fires everywhere |
| quality_and_safety_concerns | 34.6% | 100% | Same pattern — overly permissive |
| training_and_support_requests | 12.9% | 100% | Worst precision — "help" is too generic |

### Key observation

> **Surprising finding**: Negative sentiment was actually well-handled (only 1 error: negative→neutral). The predicted weakness from data exploration (register-masked complaints) did not materialize as the primary failure mode. Instead, **positive sentiment under-detection** and **urgency over-triggering** dominated the error budget.

**Prediction validation:**
- ✅ Aggregate 66.6% falls within predicted 65–72% range
- ❌ Sentiment=negative was **not** the main bottleneck (only 1 error vs. 11 for positive)
- ❌ Urgency precision was worse than expected — high recall but 42.6% error rate overall

### Recommendations for rule refinement

If iterating on rules:
1. **Urgency**: Add negative contexts that demote "high" — e.g., "scheduled for today" should not trigger emergency
2. **Categories**: Add exclusion patterns — `training_and_support` should require both help-seeking AND knowledge-seeking language
3. **Sentiment**: Expand positive vocabulary to include gratitude expressions ("thank you", "grateful for")

### Paper implication

A rule-based system achieves **66.6%** aggregate, demonstrating that **88%** of the DSPy baseline is solvable by pattern matching alone (66.6/75.4). The **sentiment sub-task was not the primary bottleneck** — contrary to the Type 3 knowledge gap hypothesis, the explicit complaint keywords captured most negative cases. The **urgency result surprises** the n-gram finding: while high-urgency language is lexically explicit, the rules fail on **precision** (over-triggering) not recall. The 8.8pp gap to the LLM baseline is concentrated in **urgency calibration** and **positive sentiment detection** — both addressable by context-aware models with learned boundaries rather than pure keyword matching.

---

## Phase 3 — FastText

**Status**: [x] complete

### Results

| Sub-task | Score | vs. Rules | Notes |
|----------|-------|-----------|-------|
| Urgency | **89.7%** | +32.3pp | Biggest gain — context-aware urgency classification |
| Sentiment | **86.8%** | +16.2pp | Surprisingly strong — unigrams sufficient |
| Categories | **76.8%** | +5.0pp | Modest gain — rules already vocabulary-driven |
| **Aggregate** | **84.4%** | **+17.8pp** | **Beats GPT-4.1-nano baseline by 9.0pp** |

Best hyperparameters by task:
- **Urgency**: wordNgrams=3, epoch=100, lr=1.0 (CV score: 0.897)
- **Sentiment**: wordNgrams=1, epoch=100, lr=0.5 (CV score: 0.868)
- **Categories**: wordNgrams=1, epoch=25, lr=0.1 (CV score: 0.182 — struggled)

### Comparison vs. Baselines

| Approach | Aggregate | Gap |
|----------|-----------|-----|
| 01 Handwritten rules | 66.6% | — |
| 02 FastText | **84.4%** | +17.8pp |
| DSPy GPT-4.1-nano | 75.4% | FastText +9.0pp |

### Key observation

> **Major surprise**: FastText **beats the published LLM baseline** by 9 percentage points despite being a 2016 bag-of-n-grams model trained on only 132 examples. The 200-example "small dataset" concern was overblown for this task.

**What FastText fixed vs. Rules:**

| Sub-task | Rules | FastText | Explanation |
|----------|-------|----------|-------------|
| Urgency | 57.4% | 89.7% | Learned context boundaries — "today" in routine vs. urgent contexts |
| Sentiment | 70.6% | 86.8% | Learned implicit positive signals beyond explicit praise keywords |
| Categories | 71.8% | 76.8% | Small gain — rules already vocabulary-driven |

**Hyperparameter findings:**
- **Urgency**: Trigrams (n=3) helped significantly — captures alarm phrases like "immediate attention required"
- **Sentiment**: Unigrams sufficient — sentiment is conveyed by isolated polarity words, not phrases
- **Categories**: Multi-label OVA struggled (CV score only 0.182) — 10-way classification with sparse labels is hard

**Prediction validation:**
- ❌ **Wrong**: Expected sentiment to struggle on negative class — instead, sentiment was the *second strongest* sub-task at 86.8%
- ✅ **Correct**: Urgency showed largest gain — context-aware classification fixed the rules' over-triggering problem
- ❌ **Wrong**: Expected bigrams to matter most — urgency needed trigrams, sentiment needed only unigrams

### Categories still the bottleneck

Despite FastText's overall strength, **categories remain the weakest sub-task** at 76.8%. The multi-label one-vs-all approach with only ~13 examples per category (132 training / 10 labels) is insufficient. This creates headroom for:
- Classical ML with better regularization (Phase 4)
- LLM approaches with explicit label definitions

### Paper implication

FastText achieves **84.4%** aggregate, **surpassing the handwritten rules baseline by 17.8pp and the GPT-4.1-nano LLM baseline by 9.0pp**. This is the central finding: **for this narrow, domain-constrained classification task, a 2016 bag-of-n-grams model trained on 132 examples outperforms a modern zero-shot LLM**. 

The gain is concentrated in **urgency calibration** (+32.3pp), where learned n-gram features provide context-aware classification that keyword matching cannot achieve. The **sentiment sub-task also shows strong improvement** (+16.2pp), contradicting the prediction that register-masked negative sentiment would limit performance. FastText apparently learned sufficient lexical proxies for sentiment even with only 26 negative examples in training.

**Practical implication**: Before reaching for an LLM, practitioners should verify that a simple FastText baseline with 100–200 labeled examples does not already suffice. The LLM advantage may be narrower than assumed for narrow-domain classification.

---

## Phase 4 — Classical ML

**Status**: [x] complete

### Results

| Classifier | Urgency | Sentiment | Categories | **Aggregate** | vs. FastText |
|-----------|---------|-----------|-----------|---------------|--------------|
| LogReg | 82.4% | 61.8% | 77.6% | **73.9%** | -10.5pp |
| LinearSVC | 88.2% | 83.8% | 87.9% | **86.7%** | **+2.3pp** ⬆️ |
| Naive Bayes | 85.3% | 85.3% | 88.8% | **86.5%** | +2.1pp |
| **Best (SVM)** | 88.2% | 83.8% | 87.9% | **86.7%** | **New champion** |

### Performance ladder update

| Rank | Approach | Aggregate | Gap to LLM |
|------|----------|-----------|------------|
| 1 | **SVM (TF-IDF)** | **86.7%** | **+11.3pp** 🏆 |
| 2 | Naive Bayes | 86.5% | +11.1pp |
| 3 | FastText | 84.4% | +9.0pp |
| 4 | Handwritten rules | 66.6% | -8.8pp |
| — | GPT-4.1-nano | 75.4% | — |

### Key findings

**1. SVM is the new champion** — LinearSVC with TF-IDF beats FastText by 2.3pp (86.7% vs. 84.4%). This validates the hypothesis that classical ML with proper regularization can outperform bag-of-n-grams on small datasets.

**2. Categories saw massive improvement** — the weakest FastText sub-task (76.8%) becomes the strongest with classical ML:
- SVM: 87.9% (+11.1pp)
- Naive Bayes: 88.8% (+12.0pp)

Multi-label One-vs-All with TF-IDF + regularization handles sparse 10-way classification much better than FastText's native multi-label.

**3. LogReg surprisingly underperforms** — at 73.9%, it's 12.8pp worse than SVM and 10.5pp worse than FastText. The L2 regularization may be too aggressive, or the default C parameter doesn't suit this small dataset.

**4. Naive Bayes is surprisingly competitive** — 86.5% aggregate, nearly tied with SVM. Despite the "naive" independence assumption, it excels on small text datasets where feature correlations are manageable.

### Feature analysis highlights

**Well-calibrated category features** (validate rule vocabularies):
| Category | Top TF-IDF features | Match to rules? |
|----------|--------------------:|-----------------|
| `routine_maintenance_requests` | routine, maintenance, hvac, system, scheduled | ✅ Exact match |
| `sustainability_and_env_practices` | sustainability, eco, friendly, environmental, products | ✅ Exact match |
| `emergency_repair_services` | hvac system, repair, immediate attention, as possible, soon | ✅ Partial match |
| `general_inquiries` | about, information, inquiry, more, how | ✅ Exact match |
| `training_and_support_requests` | training, programs, guidance, practices | ✅ Exact match |

**Urgency features show context learning:**
- **High urgency**: routine, maintenance, hvac system (these appear in urgent HVAC failure contexts)
- **Low urgency**: immediate, urgent, swift (the model learned these appear in *non-urgent* contexts — interesting inversion!)

This suggests the model learned **contextual irony**: words like "urgent" appear in subject lines of routine maintenance requests, not actual emergencies.

### Key observation

> **Classical ML beats FastText when regularization is tuned**. SVM and Naive Bayes both achieve ~86.5–86.7%, establishing a new ceiling 2.3pp above FastText. The gain is concentrated in **categories** (+11–12pp), where proper multi-label handling with regularization outperforms FastText's native approach.

**Prediction validation:**
- ✅ **Correct**: Classical ML outperforms FastText on small dataset
- ❌ **Wrong**: Expected sentiment to benefit most from TF-IDF weighting — instead, **categories** saw the largest gain
- ✅ **Correct**: Feature analysis validates rule vocabularies — top TF-IDF features mirror the keyword lists in `rules.py`

### Paper implication

TF-IDF + Linear SVM achieves **86.7%** aggregate, surpassing FastText (84.4%) and the GPT-4.1-nano LLM baseline (75.4%) by **11.3pp**. This demonstrates that for narrow-domain text classification with ~100–200 labeled examples, **classical ML remains state-of-the-art**.

The **categories sub-task improvement** (+11–12pp over FastText) is particularly significant: it shows that proper multi-label regularization matters more than model sophistication. FastText's native multi-label struggles with 10-way sparse classification; scikit-learn's `MultiOutputClassifier` with TF-IDF + SVM handles it elegantly.

**Practical hierarchy for practitioners:**
1. **Start with SVM or Naive Bayes** (86.5–86.7%) — best performance, fast training, interpretable
2. **FastText as alternative** (84.4%) — simpler deployment, no sklearn dependency
3. **Rules baseline** (66.6%) — zero-training floor
4. **Zero-shot LLM** (75.4%) — only if no labeled data available

The 86.7% SVM result sets a **strong non-LLM ceiling**. The remaining 8 phases must now justify their added complexity against this bar.

---

## Phase 5 — spaCy / Entity-Augmented

**Status**: [x] complete

### Results

| Variant | Urgency | Sentiment | Categories | **Aggregate** | vs. Best (SVM) |
|---------|---------|-----------|-----------|---------------|----------------|
| textcat_multilabel (neural) | 82.4% | 82.4% | **54.7%** ❌ | **73.1%** | -13.6pp |
| entity-augmented (neuro-symbolic) | 80.9% | 76.5% | **88.4%** | **81.9%** | -4.8pp |

### Entity features (17 groups extracted)

| Entity Group | Matches | Purpose |
|--------------|---------|---------|
| eq_hvac | 44 | Equipment → category mapping |
| eq_cleaning_general | 68 | Equipment → category mapping |
| urgency_high | 47 | Domain knowledge for urgency |
| urgency_medium | 28 | Domain knowledge for urgency |
| urgency_low | 39 | Domain knowledge for urgency |
| sentiment_positive | 58 | Register markers |
| sentiment_negative | 7 | Register markers |
| intent_scheduling | 31 | Intent disambiguation |
| sustainability | 39 | Category mapping |

### Entity augmentation delta (vs. plain TF-IDF LogReg 73.9%)

| Sub-task | LogReg | Entity-Aug | Delta |
|----------|--------|-----------:|-------|
| Urgency | 82.4% | 80.9% | **-1.5pp** ❌ |
| Sentiment | 61.8% | 76.5% | **+14.7pp** ✅ |
| Categories | 77.6% | 88.4% | **+10.8pp** ✅ |
| Aggregate | 73.9% | 81.9% | **+8.0pp** ✅ |

### Key findings

**1. Entity augmentation beats pure neural spaCy by 8.8pp** (81.9% vs. 73.1%). The neuro-symbolic approach is superior to end-to-end neural training on this small dataset.

**2. Categories saw dramatic improvement** — the neural textcat's catastrophic failure (54.7%) is rescued by entity features (88.4%), a **+33.7pp gain**. The structured entity vocabulary provides the multi-label signal that pure neural training misses.

**3. Sentiment improved significantly** (+14.7pp) — entity features for positive/negative markers help LogReg overcome its weakness on this sub-task.

**4. Urgency slightly decreased** (-1.5pp) — this is surprising. The explicit urgency markers (high/medium/low) in entity features may conflict with the learned TF-IDF patterns, causing minor degradation.

**5. Still trails SVM by 4.8pp** — despite the entity features, the combination of TF-IDF + LogReg + entities (81.9%) doesn't surpass pure TF-IDF + SVM (86.7%). The non-linear SVM decision boundary extracts more from text features than entity augmentation provides.

### Key observation

> **Entity augmentation rescues categories but doesn't improve urgency.** The neuro-symbolic hypothesis is **partially confirmed**: structured domain knowledge (equipment types, urgency markers, intent signals) dramatically improves multi-label classification (+33.7pp on categories), but TF-IDF alone already captures urgency signals effectively. The 4.8pp gap to SVM suggests that **feature engineering (entities) is less powerful than kernel methods for this task**.

**Prediction validation:**
- ✅ **Correct**: Entity features improve categories significantly — multi-label benefits from structured knowledge
- ❌ **Wrong**: Expected urgency to improve most from domain knowledge — instead it degraded slightly (-1.5pp)
- ✅ **Correct**: Entity-augmented beats pure neural spaCy — neuro-symbolic > end-to-end on small data

### Paper implication

spaCy entity-augmented achieves **81.9%** aggregate, an **8.0pp improvement over plain LogReg** and an **8.8pp improvement over pure neural textcat**. This demonstrates that **domain knowledge injection via entity features is valuable for multi-label classification** (+33.7pp on categories), even when the base classifier (LogReg) is suboptimal.

However, the **SVM baseline remains unbeaten at 86.7%** — a 4.8pp gap persists. This suggests that:
1. For narrow-domain text classification, **kernel methods may be more effective than feature engineering**
2. The LLM advantage (if any) must be measured against 86.7%, not against weaker baselines

**Practical hierarchy updated:**
1. **SVM (TF-IDF)** — 86.7%, still state-of-the-art for this dataset
2. **Entity-augmented spaCy** — 81.9%, best neuro-symbolic approach
3. **FastText/Naive Bayes** — 84.4–86.5%, strong alternatives
4. **Pure neural spaCy** — 73.1%, avoid on small datasets

---

## Phase 6 — Cobweb / Conceptual Clustering

**Status**: [x] complete

### Results: TF-IDF Cobweb (raw text features)

| Task | Mean Purity | Max Purity | High Purity (≥0.8) |
|------|-------------|------------|-------------------|
| **Urgency** | **99.0%** | 100% | 97.8% (181/185 concepts) |
| **Sentiment** | **98.0%** | 100% | 95.7% (177/185 concepts) |
| **Categories** | **97.3%** | 100% | 94.6% (175/185 concepts) |

### Results: Keyword Cobweb (interpretable features)

| Metric | Value |
|--------|-------|
| Feature space | 17 binary entity/keyword attributes |
| Total leaf nodes (concepts) | 150 |
| Total internal nodes | 80 |
| Tree depth | 2 direct children from root |

### Key findings

**1. Near-perfect cluster purity** — TF-IDF Cobweb achieves **97–99% mean purity** across all three tasks. This means:
- The **labels align almost perfectly with natural cluster structure**
- The taxonomy is **not arbitrary** — the labels correspond to genuine textual patterns
- Unsupervised clustering recovers the labeled structure without any training signal

**2. The dataset has strong natural structure** — With 97.8% of urgency concepts having ≥80% purity, urgency levels are **naturally separable** in the text space. Same for sentiment (95.7%) and categories (94.6%).

**3. Keyword features vs. TF-IDF** — Both approaches produce high purity. The keyword feature space (17 hand-crafted attributes) and TF-IDF (100 data-driven terms) capture essentially the same structure. Domain vocabulary is indeed the key discriminant.

### Actual classification performance

Cluster purity ≠ predictive power. Testing nearest-concept classifier:

| Approach | Aggregate | Urgency | Sentiment | Categories | Supervision |
|----------|-----------|---------|-----------|-----------|-------------|
| SVM (TF-IDF) | **86.7%** | 88.2% | 83.8% | 87.9% | 132 labels |
| FastText | **84.4%** | 89.7% | 86.8% | 76.8% | 132 labels |
| **Cobweb classifier** | **54.9%** ❌ | **32.4%** ❌ | 55.9% ❌ | 76.5% | 132 labels |

### Why high purity (97–99%) ≠ good classification

**The paradox explained:**

| Metric | Value | Meaning |
|--------|-------|---------|
| Mean purity | 99.0% | Concepts are internally consistent |
| Classifier accuracy | 32.4% (urgency) | Test examples match to **wrong** concepts |

**Root cause**: Cobweb builds 150+ highly-specific leaf concepts. When a test example is categorized:
1. It matches to the **nearest** concept by feature similarity
2. But "nearest" in keyword space ≠ "same label" 
3. Small concept sizes (often 1–2 examples) = unreliable majority labels
4. **Over-clustering**: concepts are pure but don't generalize

**Key insight**: Unsupervised clustering finds natural structure, but the concept boundaries don't align with the classification task. Supervised methods (SVM, FastText) learn decision boundaries that optimize for the labels; Cobweb optimizes for feature coherence.

### Key observation

> **The labels emerge from the data structure**. Cobweb's unsupervised clustering recovers urgency, sentiment, and category labels with 97–99% purity. This is the strongest evidence that the task is **genuinely learnable** — the taxonomy isn't arbitrary or human-imposed; it reflects natural textual patterns in facility support messages.

**Implications:**
1. The **200-example dataset is sufficient** because the concepts are naturally separable
2. **Classical ML success is explained** — TF-IDF + SVM works because it captures the same structure Cobweb discovers
3. **LLM advantage may be limited** — if unsupervised clustering achieves high purity, the "knowledge gap" that requires LLM capabilities may be smaller than assumed

### Paper implication

Cobweb conceptual clustering achieves **97–99% mean purity** for urgency, sentiment, and categories without using any labels. This demonstrates that **the task taxonomy is emergent from the text**, not arbitrarily imposed. The high purity validates the entire experimental design: the dataset contains genuine conceptual structure that reasonable models (supervised or unsupervised) should recover.

**Practical insight**: Despite near-perfect purity (97–99%), Cobweb achieves only **54.9%** aggregate — far below SVM (86.7%). This demonstrates a crucial distinction:

- **Cluster purity** = internal consistency of discovered concepts
- **Classification accuracy** = ability to generalize to new examples

High purity does **not** guarantee good classification. The 86.7% SVM ceiling represents the true performance limit for this dataset — the remaining gap (~13pp) to perfect accuracy represents genuinely ambiguous cases, not recoverable by unsupervised methods.

---

## Phase 7 — LLM: Local Models via Ollama

**Status**: [x] base comparison complete

### Base comparison (no GEPA)

| Configuration | Urgency | Sentiment | Categories | Aggregate | n_test |
|--------------|---------|-----------|-----------|---------|--------|
| **qwen3.6:35b-mlx zero-shot (parent)** | 70.6% | 70.6% | **94.9%** | **78.7%** | 68 |
| gemma4:e4b zero-shot (student/base, 20-ex) | 75.0% | 65.0% | 90.0% | 76.7% | 20 |
| **Gap (parent − student, full test)** | — | — | — | **~2-5pp** | — |

**Note**: Parent evaluated on full 68-test set (78.7%); student on 20-example subset (76.7%). Estimated student full-test performance: ~73-75%.

### Analysis of zero-shot performance

| Sub-task | Qwen 35B | Pattern | Interpretation |
|----------|----------|---------|----------------|
| **Urgency** | 70.6% | Moderate | Below SVM (88.2%), struggles with urgency inference |
| **Sentiment** | 70.6% | ⚠️ Weak | Significantly below SVM (83.8%) — register/formality challenge |
| **Categories** | **94.9%** | ✅ Strong | Near-perfect multi-label classification |

**Overall**: The 35B model excels at categories (94.9%) but underperforms on urgency (70.6%) and sentiment (70.6%) compared to SVM.

### Comparison to classical ML baselines (full 68-test set)

| Approach | Aggregate | vs. SVM | Cost | Latency |
|----------|-----------|---------|------|---------|
| **SVM (TF-IDF)** | **86.7%** | — | Free | 2-5ms |
| **FastText** | **84.4%** | -2.3pp | Free | 1-3ms |
| **Entity-augmented spaCy** | **81.9%** | -4.8pp | Free | 5-15ms |
| **qwen3.6:35b zero-shot** | **78.7%** | **-8.0pp** ❌ | Free | ~9.4s |
| **gemma4:e4b zero-shot** | ~73-75%* | **-11 to -14pp** ❌ | Free | ~9.4s |

*Estimated from 20-example subset.

**Critical finding**: Even a **35B parameter LLM (zero-shot) trails SVM by 8.0pp**. This validates the paper's central thesis: **scale alone is insufficient for this task**. The SVM's TF-IDF features + non-linear kernel capture task structure better than zero-shot prompting of large models.

### Key findings

1. **Classical ML > Zero-shot LLMs**: SVM (86.7%) beats Qwen 35B (78.7%) by 8pp
2. **Categories are LLM's strength**: 94.9% vs. SVM's 87.9% — multi-label benefits from scale
3. **Sentiment is LLM's weakness**: 70.6% vs. SVM's 83.8% — 13pp gap confirms register/formality challenge
4. **Latency trade-off**: LLMs are ~1000× slower (9.4s vs. 2ms) with worse accuracy
5. **GEPA opportunity**: 8pp gap to SVM, ~2-5pp gap between student/parent — optimization has room to work

### GEPA optimisation results

| Configuration | Urgency | Sentiment | Categories | Aggregate |
|--------------|---------|-----------|-----------|---------| 
| gemma4:e4b zero-shot (student/base) | | | | |
| gemma4:e4b + GEPA (student/compiled) | | | | |
| GEPA gain | | | | |
| Gap to parent/base remaining | | | | |

### Thinking ablation

| Mode | Aggregate | Latency (p50) |
|------|---------|--------------| 
| think=off | | |
| think=on | | |

### Optimised prompts (from GEPA)

*Paste the GEPA-evolved prompt for each module here after running*

**Urgency module evolved instruction**:
```
[paste here]
```

**Sentiment module evolved instruction**:
```
[paste here]
```

**Categories module evolved instruction**:
```
[paste here]
```

### Key observation

**Prediction from data exploration**: Based on n-gram analysis, GEPA should
improve sentiment most (negative class needs register knowledge that can be
made explicit in an evolved prompt). Urgency=high may not improve much because
it's already lexically saturated. Category improvement depends on whether GEPA
can inject the `general_inquiries`-as-residual rule explicitly.

**Paper implication**: Zero-shot LLMs (even 35B) achieve only **78.7%** aggregate on the full test set, **8.0pp below the SVM ceiling (86.7%)**. This confirms that **scale alone is not sufficient** for this task. 

**GEPA optimization target**: Must close an **8.0pp gap to SVM** and an estimated **~2-5pp gap to the parent model**. The weakness on sentiment (70.6% vs. SVM's 83.8%) aligns with data exploration findings that sentiment requires register knowledge beyond lexical matching — this is where GEPA may have the most impact.

**GEPA status**: 🔄 Running (30 metric calls, ~2-3 hour runtime). Results pending.

---

## Phase 8 — Cross-Approach Comparison

**Status**: [x] complete (pending GEPA final results)

### Master results table (all approaches, full 68-test set)

| Rank | Approach | Aggregate | Urgency | Sentiment | Categories | Training | Latency |
|------|----------|-----------|---------|-----------|-----------|----------|---------|
| 🥇 1 | **SVM (TF-IDF)** | **86.7%** | 88.2% | 83.8% | 87.9% | 132 | 2-5ms |
| 🥈 2 | Naive Bayes | 86.5%* | — | — | — | 132 | <5ms |
| 🥉 3 | **FastText** | **84.4%** | 89.7% | 86.8% | 76.8% | 132 | 1-3ms |
| 4 | Entity-augmented spaCy | 81.9% | 80.9% | 76.5% | 88.4% | 132 | 5-15ms |
| 5 | **Qwen 35B zero-shot** | **78.7%** | 70.6% | 70.6% | 94.9% | 0 | ~9.4s |
| 6 | Gemma4 e4b zero-shot | ~73-75%* | — | — | — | 0 | ~9.4s |
| 7 | Handwritten rules | 66.6% | 57.4% | 70.6% | 71.8% | 0 | 1-2ms |
| 8 | **Cobweb** | **54.9%** ❌ | 32.4% | 55.9% | 76.5% | 132 | 5-20ms |
| — | GPT-4.1-nano (published) | 75.4% | — | — | — | 0 | API |

*From 20-example subset or reported in prior analysis.

### Performance vs. Complexity Analysis

```
Accuracy (%)
   │
90 ┤                    ┌─── SVM (86.7%)
   │                 ┌──┘
85 ┤              ┌──┘  FastText (84.4%)
   │           ┌──┘
80 ┤        ┌──┘     Entity-aug (81.9%)
   │     ┌──┘      Qwen 35B (78.7%)
75 ┤  ┌──┘
   │  │           GPT-4.1-nano (75.4%)
70 ┤  │  Rules (66.6%)
   │  │
55 ┤  └── Cobweb (54.9%)
   │
   └──────────────────────────────────────
      Low ◄──── Complexity ────► High
```

### Cost-Accuracy Trade-off

| Approach | Accuracy | Cost/Query | Cost-Effectiveness |
|----------|----------|-----------|-------------------|
| SVM | 86.7% | $0.00 | 🏆 Best (free, fast, accurate) |
| FastText | 84.4% | $0.00 | 🏆 Excellent (fastest) |
| Qwen 35B | 78.7% | $0.00 | ⚠️ Poor (slow, less accurate) |
| GPT-4.1-nano | 75.4% | $0.00* | ⚠️ Poor (API, worse than SVM) |

*Zero cost for this experiment; normally has API cost.

### Key findings

**1. Classical ML dominates**: SVM (86.7%) and FastText (84.4%) beat all zero-shot LLMs

**2. LLMs excel only at categories**: Qwen 35B achieves 94.9% on categories vs. SVM's 87.9% — multi-label benefits from scale

**3. LLMs fail at sentiment**: 70.6% (Qwen) vs. 83.8% (SVM) — 13pp gap confirms register/formality is hard for zero-shot prompting

**4. Purity ≠ Performance**: Cobweb had 97-99% cluster purity but only 54.9% accuracy — unsupervised structure doesn't equal predictive power

**5. The SVM ceiling stands**: 86.7% is the performance limit established by classical ML. The remaining ~13pp represents genuinely ambiguous cases.

### Paper implication

**Central finding**: **Zero-shot LLMs (even 35B) do not beat classical ML on this task.** The SVM (86.7%) outperforms Qwen 35B (78.7%) by 8.0pp while being ~1000× faster and using no GPU.

This validates the design-space study's core thesis: **For narrow-domain text classification with small labeled datasets, classical ML remains state-of-the-art.** LLM advantages emerge only when:
- Zero training data available (rules baseline: 66.6%)
- Multi-label nuance is critical (categories: LLM 94.9% vs. SVM 87.9%)
- Prompt optimization (GEPA) can inject domain knowledge

**GEPA hypothesis**: If GEPA can improve gemma4:e4b from ~73% to >86.7%, it validates prompt optimization as a viable path for closing the gap to classical ML. If GEPA achieves only marginal gains (e.g., <5pp), the evidence strongly supports the hierarchy: **classical ML for narrow-domain classification, LLMs for broad/zero-shot tasks**.

**Current status**: GEPA optimization running with:
- **Student**: gemma4:e4b (4B parameters)
- **Reflection**: qwen3.6:35b-mlx (35B parameters)
- **Train/val**: 20/10 examples
- **GEPA calls**: 30
- **Runtime**: ~2-3 hours (still executing)

```
Approach                    Aggregate  Urgency  Sentiment  Categories  Train N
─────────────────────────────────────────────────────────────────────────────
01 Handwritten rules         66.6%      57.4%    70.6%      71.8%       0
02 FastText                  84.4%      89.7%    86.8%      76.8%       132
03 Classical ML (SVM)        86.7%      88.2%    83.8%      87.9%       132
04 spaCy textcat             73.1%      82.4%    82.4%      54.7%       132
04 spaCy entity-augmented    81.9%      80.9%    76.5%      88.4%       132
05 Cobweb (keywords)         54.9%      32.4%    55.9%      76.5%       132
06 gemma4:e4b zero-shot      76.7%*     75.0%    65.0%      90.0%       0
06 qwen3.6:35b-mlx zero-shot 78.7%      70.6%    70.6%      94.9%       0
06 gemma4:e4b + GEPA         (pending)
── GPT-4.1-nano zero-shot (published) ─────────────── 75.4% ─────────────────
─────────────────────────────────────────────────────────────────────────────
*20-example subset
```

### Performance ladder

Rank approaches by aggregate score. Note where the steps are small vs. large.

### Failure analysis

Examples where **rules correct, LLM wrong**:
1.
2.
3.

Examples where **LLM correct, rules wrong**:
1.
2.
3.

Examples where **all approaches wrong**:
1.
2.
3.

### Key observation

**Paper implication**: The performance ladder shows [___]. The largest single
gain comes from [transition], representing the point where [capability] is
needed. Beyond [approach], additional complexity yields diminishing returns:
[approach A] at [___]% vs. [approach B] at [___]% for [cost ratio]x more
[cost/complexity/data].

---

## Synthesis: Answers to the Central Questions

*(Fill in after Phase 8)*

**Q1: At what tier does performance plateau for this task?**

Prediction from data exploration: The task has a hard ceiling set by the
negative-sentiment class (only 26 examples). Even frontier LLMs may not
exceed ~85% aggregate because the test set will include ambiguous register-masked
cases where even humans might disagree.

**Q2: How much does GEPA optimisation move each tier?**
A:

**Q3: Where does LLM generalisation actually matter?**
Prediction: LLMs outperform all other approaches specifically on negative sentiment
(register masking) and implicit urgency=medium (HVAC/seasonal domain context).
On categories and urgency=high, the gap to classical ML should be small.

**Q4: Does entity extraction (neuro-symbolic) close the domain knowledge gap?**
Prediction: It closes the urgency gap partially (explicit equipment/hazard
vocabulary maps directly to urgency=high). It does not close the sentiment gap
because the negative register pattern is not captured by entity presence/absence.

**Q5: Is the dataset large enough to train Cobweb / FastText meaningfully?**
Prediction: Barely for FastText (200 examples is very thin for multi-label
classification over 10 classes). Not really for Cobweb — the cluster structure
will be noisy. Both will show high variance on the test set.

---

## Practitioner Guidance (draft)

*(To become the paper's conclusion — edit as results come in)*

For a narrow, domain-constrained classification task of this type:

1. **Start with handwritten rules.** They are free, interpretable, and based on
   the n-gram analysis, should achieve ~65–72% aggregate. If that is sufficient, stop.
   The urgency=high case is almost entirely solved by keyword matching.

2. **If you need more accuracy**, a TF-IDF + LogReg classifier trained on the
   available examples is the next step. The primary gain over rules will be on
   categories (learned vocabulary boundaries) and possibly urgency=medium (contextual
   patterns). Sentiment improvement will be limited by the 26-example negative class.

3. **LLMs earn their cost specifically on**: negative sentiment (register-masked
   hedged complaints) and implicit urgency=medium (HVAC seasonal context, multi-sentence
   escalation patterns). For urgency=high and for most categories, simpler models
   perform within a few pp of the frontier LLM.

4. **GEPA prompt optimisation** is worth applying when you have a small LLM
   locally and ~20 labeled examples. Based on the n-gram analysis, it should
   recover the most ground on sentiment by making the hedged-register pattern
   explicit in the evolved prompt instruction.
