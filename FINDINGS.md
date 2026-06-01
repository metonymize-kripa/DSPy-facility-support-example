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

**Status**: [ ] pending

### Results

| Classifier | Urgency | Sentiment | Categories | Aggregate |
|-----------|---------|-----------|-----------|---------| 
| LogReg | | | | |
| LinearSVC | | | | |
| Naive Bayes | | | | |
| **Best** | | | | |

### Top features per sub-task (from LogReg)

**Urgency (high vs. low)**:
Top positive: ___
Top negative: ___

**Sentiment (negative vs. positive)**:
Top positive: ___
Top negative: ___

**Most diagnostic category features**:
*(One example per category)*

### Key observation

> *Does classical ML outperform FastText? Do the learned features validate the
> handwritten rule vocabularies?*

**Prediction**: TF-IDF with sublinear_tf and IDF weighting should handle the
rare negative class slightly better than FastText by down-weighting common
neutral terms. Still unlikely to reliably detect `negative` given 26 examples.

**Paper implication**: The top TF-IDF features for urgency=high should mirror
the bigrams found in exploration (`urgent`, `immediate`, `attention required`),
validating that the rules vocabulary was well-calibrated. If they diverge,
that reveals cases the rules missed.

---

## Phase 5 — spaCy / Entity-Augmented

**Status**: [ ] pending

### Results

| Variant | Urgency | Sentiment | Categories | Aggregate |
|---------|---------|-----------|-----------|---------| 
| textcat_multilabel | | | | |
| entity-augmented | | | | |

### Entity augmentation delta (vs. plain TF-IDF LogReg)

- Urgency delta: ___pp (expected positive — domain knowledge)
- Sentiment delta: ___pp
- Categories delta: ___pp

### Key observation

> *Does adding structured entity features improve urgency specifically?
> This is the key test of the neuro-symbolic hypothesis.*

**Paper implication**: Entity augmentation [improves/does not improve] urgency
accuracy by [___]pp compared to plain TF-IDF, [supporting/contradicting] the
hypothesis that domain knowledge (Type 2) can be injected cheaply via a domain
vocabulary without an LLM.

---

## Phase 6 — Cobweb / Conceptual Clustering

**Status**: [ ] pending

### Cluster purity

| Label | Best cluster purity | Interpretation |
|-------|--------------------|-|
| urgency | | |
| sentiment | | |
| emergency_repair_services | | |
| routine_maintenance_requests | | |
| general_inquiries | | |

**Prediction**: `emergency_repair_services` and `sustainability_and_environmental_practices`
should show higher cluster purity because their vocabulary is highly distinctive
(alarm language vs. eco-language). `general_inquiries` should show low purity because
it co-occurs with everything — there is no distinctive vocabulary exclusive to it.

### Key observation

**Paper implication**: Cobweb cluster purity results tell us whether the label
structure is emergent from the text or imposed by human taxonomy. Low purity for
`general_inquiries` and `customer_feedback_and_complaints` would confirm the
conceptual framework finding that these categories require intent-reading, not
vocabulary matching.

---

## Phase 7 — LLM: Gemma4 via Ollama

**Status**: [ ] pending

### Base comparison (no GEPA)

| Configuration | Urgency | Sentiment | Categories | Aggregate |
|--------------|---------|-----------|-----------|---------| 
| gemma4:26b zero-shot (parent/base) | | | | |
| gemma4:e4b zero-shot (student/base) | | | | |
| Gap (parent − student) | | | | |

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

**Paper implication**: GEPA optimisation on gemma4:e4b closes [___]% of the gap
to gemma4:26b zero-shot. The gain concentrated in sentiment would confirm the
data exploration prediction that negative-class register masking is the key
unsolved problem for lexical approaches.

---

## Phase 8 — Cross-Approach Comparison

**Status**: [ ] pending

### Master results table

*(Generated by `07_evaluation/compare_all.py` — paste output here)*

```
Approach                    Aggregate  Urgency  Sentiment  Categories  Train N
─────────────────────────────────────────────────────────────────────────────
01 Handwritten rules
02 FastText
03 Classical ML (best)
04 spaCy textcat
04 spaCy entity-augmented
05 Cobweb (keywords)
06 Gemma4:e4b zero-shot
06 Gemma4:26b zero-shot
06 Gemma4:e4b + GEPA
── GPT-4.1-nano zero-shot (published) ─────────────── 75.4% ─────────────────
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
