# Engineering Guide: Approach Selection for Facility Support Classification

**A practitioner's framework for evaluating NLP classification approaches across technical, economic, and operational dimensions.**

---

## Executive Summary

This guide complements the empirical results in `FINDINGS.md` with an engineering analysis of **what it takes to build, deploy, and maintain** each approach. We compare seven methods across four dimensions:

| Dimension | Key Question |
|-----------|--------------|
| **Knowledge Injection** | Where does domain expertise enter the system? |
| **Effort & Economics** | Pre-training, training, inference, and maintenance costs |
| **Operational Characteristics** | Maintainability, reliability, customization needs |
| **Generalization Trade-offs** | What you gain vs. what you sacrifice |

**Bottom line**: For narrow-domain text classification with small labeled datasets, classical ML (SVM, FastText) dominates on accuracy, cost, and maintainability. LLMs show value only in specific scenarios: zero-shot settings, multi-label nuance, or when GEPA optimization can close the accuracy gap.

---

## 1. Approach Overview Matrix

| Approach | Aggregate | Pre-Training Effort | Training Effort | Inference Cost | Maintainability |
|----------|-----------|---------------------|-----------------|----------------|-----------------|
| **Handwritten Rules** | 66.6% | Very Low | None | Negligible | Poor |
| **FastText** | 84.4% | None | Low | Negligible | Good |
| **SVM (TF-IDF)** | 86.7% | None | Low | Negligible | Good |
| **spaCy textcat** | 73.1% | None | Medium | Low | Moderate |
| **Entity-Augmented spaCy** | 81.9% | **High** (entity vocab) | Medium | Low | **Poor** |
| **gemma4:e4b zero-shot** | 75.4% | **Massive** (4B params) | None | **High** | **Excellent** |
| **qwen3.6:35b zero-shot** | 78.7% | **Massive** (35B params) | None | **Very High** | **Excellent** |
| **gemma4:e4b + GEPA** | (pending) | Massive | **High** (prompt opt) | High | Moderate |

---

## 2. Detailed Analysis by Approach

### 2.1 Handwritten Rules (66.6%)

**Knowledge Injection**: Direct manual encoding of domain expertise via regex patterns and keyword lists.

| Aspect | Assessment |
|--------|------------|
| **Pre-training** | None |
| **Training** | None |
| **Inference** | Regex matching (sub-millisecond) |
| **Know-how source** | Human domain expert writes rules |
| **Manual judgment** | **Critical** — requires deep domain knowledge to craft effective patterns |
| **Maintenance** | **Difficult** — rules proliferate, edge cases accumulate, no systematic way to update |
| **Customization** | High per-deployment effort |
| **Generalization** | **Poor** — fails on paraphrasing, novel phrasing, edge cases |

**When to use**: 
- Zero training data available
- Extremely high interpretability required
- Temporary prototype before ML solution
- Regulatory environment requiring explainable decisions

**When to avoid**:
- Any labeled data available (>50 examples)
- Changing vocabulary or domain
- Multi-label classification needed

---

### 2.2 FastText (84.4%)

**Knowledge Injection**: Learned from training data via bag-of-n-grams representation.

| Aspect | Assessment |
|--------|------------|
| **Pre-training** | None (or use pre-trained embeddings optionally) |
| **Training** | ~minutes on CPU; automatic hyperparameter search |
| **Inference** | Sub-millisecond per example |
| **Know-how source** | Training labels only — no manual feature engineering |
| **Manual judgment** | Minimal — select hyperparameters via cross-validation |
| **Maintenance** | **Easy** — retrain on new data, no feature engineering |
| **Customization** | Minimal — hyperparameters transfer across similar domains |
| **Generalization** | **Good within domain** — struggles with out-of-vocabulary terms |

**Key Engineering Advantage**: FastText achieves **84.4%** (within 2.3pp of SVM) with minimal engineering effort. The n-gram representation captures multi-word expressions automatically.

**Deployment Characteristics**:
- Single binary model file
- No dependencies beyond FastText library
- CPU-only inference
- Easily containerized

**When to use**:
- Need quick, robust baseline
- Deployment environment constrained (mobile, edge)
- Domain vocabulary is stable

**When to avoid**:
- Heavy paraphrasing or semantic nuance
- Requires interpretability

---

### 2.3 SVM with TF-IDF (86.7%)

**Knowledge Injection**: Learned from training data via TF-IDF weighted term frequencies.

| Aspect | Assessment |
|--------|------------|
| **Pre-training** | None |
| **Training** | ~seconds on CPU; grid search over C, kernel, n-grams |
| **Inference** | Sub-millisecond per example |
| **Know-how source** | Training labels only |
| **Manual judgment** | Minimal — hyperparameter tuning via CV |
| **Maintenance** | **Excellent** — scikit-learn stability, easy retraining |
| **Customization** | Minimal — TF-IDF + SVM is domain-agnostic |
| **Generalization** | **Excellent within domain** — non-linear kernel captures complex boundaries |

**Key Engineering Advantage**: The **state-of-the-art champion (86.7%)** with the best cost-effectiveness ratio. TF-IDF features are interpretable, and the linear/non-linear kernel adapts to task complexity.

**Deployment Characteristics**:
- Pickle/Joblib serialization
- Pure Python + scikit-learn
- No GPU required
- Mature ecosystem, extensive documentation

**When to use**:
- **Default choice** for narrow-domain text classification
- Production systems requiring reliability
- Interpretability needed (feature weights visible)

**When to avoid**:
- Zero training data (use rules or zero-shot LLM)
- Real-time adaptation required (need online learning)
- Heavy semantic paraphrasing (use LLM)

---

### 2.4 spaCy textcat (73.1%)

**Knowledge Injection**: Neural network learns from training data via token embeddings.

| Aspect | Assessment |
|--------|------------|
| **Pre-training** | en_core_web_sm embeddings (optional) |
| **Training** | ~minutes on GPU; 20 epochs, early stopping |
| **Inference** | ~5-15ms per example (CPU) |
| **Know-how source** | Training labels + pre-trained embeddings |
| **Manual judgment** | Moderate — architecture selection, hyperparameters |
| **Maintenance** | **Difficult** — neural models brittle to data drift |
| **Customization** | High — requires architecture tuning per domain |
| **Generalization** | **Poor on small data** — overfits easily |

**Key Engineering Issue**: Achieves only **73.1%** — worse than classical ML despite neural architecture. **Small dataset (132 examples) insufficient** for neural training. Dev AUC of 0.0 for categories indicates complete training failure.

**When to use**:
- Large labeled dataset available (>1000 examples)
- Need differentiable pipeline for end-to-end training
- Already using spaCy ecosystem

**When to avoid**:
- Small dataset (<500 examples) — use classical ML instead
- Need reliable, reproducible results

---

### 2.5 Entity-Augmented spaCy (81.9%)

**Knowledge Injection**: **Manual** entity vocabulary + learned TF-IDF + LogReg.

| Aspect | Assessment |
|--------|------------|
| **Pre-training** | **High** — hand-craft 17 entity groups with domain terms |
| **Training** | Minutes on CPU |
| **Inference** | ~5-15ms (spaCy NER + LogReg) |
| **Know-how source** | **Human expert defines entity vocabulary** + training labels |
| **Manual judgment** | **High** — requires domain knowledge to identify discriminative entity types |
| **Maintenance** | **Poor** — entity vocabulary must be updated as domain evolves |
| **Customization** | **Very high** — per-deployment entity engineering |
| **Generalization** | Moderate — entity vocabulary may not transfer |

**Key Engineering Trade-off**: +8.8pp improvement over pure neural spaCy, but **requires manual entity engineering**. Still trails SVM by 4.8pp despite the effort.

**Maintenance Burden**:
- Entity vocabulary drifts as product/services change
- New equipment types require manual vocabulary updates
- No automatic learning of new entity patterns

**When to use**:
- Domain has stable, well-defined entity types
- Need neuro-symbolic interpretability
- Already invested in spaCy ecosystem

**When to avoid**:
- Domain vocabulary changes frequently
- Want low-maintenance solution (use pure SVM instead)

---

### 2.6 Zero-Shot LLMs (gemma4:e4b, qwen3.6:35b)

**Knowledge Injection**: **Massive pre-training** on internet-scale text — no task-specific training.

| Aspect | Assessment |
|--------|------------|
| **Pre-training** | **Massive** — 4B-35B parameters trained on trillions of tokens |
| **Training** | None (zero-shot) |
| **Inference** | **High** — 9.4s/example (GPU-intensive) |
| **Know-how source** | Pre-training data only — no task-specific knowledge |
| **Manual judgment** | **Very low** — just write a prompt |
| **Maintenance** | **Excellent** — no retraining needed |
| **Customization** | **Low** — prompt engineering only |
| **Generalization** | **Excellent across domains** — but underperforms on specific task (78.7% vs. SVM 86.7%) |

**Key Engineering Finding**: Despite **35B parameters**, Qwen achieves only **78.7%** — **8pp below SVM**. The "general knowledge" doesn't translate to task-specific performance.

**Economic Reality**:
- 35B model requires ~30GB GPU memory
- 9.4s latency vs. 2ms for SVM = **4700× slower**
- Zero training cost, but **massive inference cost** at scale

**When to use**:
- **Zero training data** available (rules baseline: 66.6%)
- Need rapid prototyping without data collection
- Task requires broad world knowledge beyond domain text

**When to avoid**:
- Labeled data available (>100 examples) — classical ML wins
- Latency-sensitive application
- Cost-sensitive deployment at scale

---

### 2.7 GEPA-Optimized LLM (gemma4:e4b + GEPA) - Pending

**Knowledge Injection**: Pre-training + **prompt optimization** using reflection model.

| Aspect | Assessment |
|--------|------------|
| **Pre-training** | Massive (student model) |
| **Training** | **High** — GEPA optimization (hours of reflection model inference) |
| **Inference** | High (same as zero-shot) |
| **Know-how source** | Pre-training + **optimized prompts** + reflection model feedback |
| **Manual judgment** | Low — automated genetic algorithm optimization |
| **Maintenance** | Moderate — prompts are code artifacts to version control |
| **Customization** | Moderate — GEPA adapts prompts to domain automatically |
| **Generalization** | Unknown — depends on GEPA's ability to generalize from small optimization set |

**Open Question**: Can GEPA close the **8pp gap** to SVM? If yes, it validates prompt optimization as viable. If no, classical ML remains champion.

**When to use**:
- Have unlabeled data + larger teacher model (Qwen 35B)
- Want LLM flexibility with better task performance
- Can afford optimization compute cost

**When to avoid**:
- Optimization cost exceeds value (use zero-shot or classical ML)
- Need immediate deployment (optimization takes hours)

---

## 3. Comparative Analysis: Knowledge Sources

| Approach | Knowledge Source | Engineering Implication |
|----------|-----------------|------------------------|
| Rules | Human expert manual encoding | Brittle, expertise-dependent, hard to maintain |
| FastText | Training data patterns | Automatic, requires only labels |
| SVM | Training data + kernel | Automatic, interpretable features |
| spaCy textcat | Training data + pre-trained embeddings | Needs large data to overcome small dataset limitations |
| Entity-Augmented | **Human vocab + training data** | **Hybrid: manual effort + learning** |
| Zero-shot LLM | Massive pre-training only | No task-specific knowledge, generic performance |
| GEPA-optimized | Pre-training + **learned prompts** | **Meta-learning: algorithm extracts task knowledge** |

**Key Insight**: The knowledge source determines maintainability:
- **Manual (Rules, Entity-Augmented)**: High initial effort, poor maintenance
- **Automatic (FastText, SVM)**: Low effort, good maintenance
- **Pre-trained (LLM)**: No task effort, but generic performance
- **Meta-learned (GEPA)**: Optimization effort, but transferable

---

## 4. Economic Analysis

### 4.1 Cost Structure Comparison

| Approach | Setup Cost | Per-Query Cost | Maintenance Cost | Total Cost of Ownership (1M queries) |
|----------|-----------|----------------|------------------|--------------------------------------|
| Rules | Low | Negligible | **High** (expert time) | **High** |
| FastText | Low | Negligible | Low | **Low** |
| SVM | Low | Negligible | Low | **Low** |
| Entity-Augmented | **Medium** (vocab) | Low | **Medium** | Medium |
| LLM Zero-shot | None | **High** (GPU) | None | **Very High** |
| GEPA | **High** (optimization) | High | Medium | High |

### 4.2 Latency vs. Accuracy Trade-off

```
Latency (log scale)
    │
10s ┤                           LLM (9.4s, 78.7%)
    │
 1s ┤
    │
10ms┤               Entity-aug (10ms, 81.9%)
    │
 1ms┤    Rules (1ms, 66.6%)   FastText (1ms, 84.4%)   SVM (2ms, 86.7%)
    │
    └────────────────────────────────────────
       60%    70%    80%    90%   Accuracy
                            ↑ SVM champion (86.7%)
```

**Critical Finding**: SVM achieves **best accuracy at lowest latency**. No trade-off required.

---

## 5. Reliability & Risk Assessment

### 5.1 Failure Modes

| Approach | Common Failures | Mitigation |
|----------|-----------------|------------|
| Rules | Paraphrasing, novel terms | Regular rule updates (high effort) |
| FastText | OOV terms, semantic nuance | Subword embeddings help, but limited |
| SVM | Feature sparsity on rare classes | Class weighting, more data |
| spaCy textcat | Overfitting, training instability | Larger dataset, regularization |
| Entity-Augmented | Entity vocabulary drift | Manual vocabulary maintenance |
| LLM Zero-shot | Hallucination, inconsistent formatting | Prompt engineering, output validation |
| GEPA | Optimization failure, prompt brittleness | Multiple seeds, validation checks |

### 5.2 Operational Risk

| Approach | Risk Level | Key Risk |
|----------|-----------|----------|
| Rules | **High** | Expert leaves, knowledge lost |
| FastText | Low | Model drift, periodic retraining |
| SVM | **Very Low** | Mature library, stable behavior |
| spaCy textcat | High | Neural brittleness, version dependencies |
| Entity-Augmented | **High** | Vocabulary maintenance burden |
| LLM Zero-shot | Moderate | API/model availability, latency spikes |
| GEPA | Moderate | Prompt optimization may not converge |

---

## 6. Decision Framework

### 6.1 Selection Flowchart

```
Start
  │
  ├──► Have labeled data? ──Yes──► Use SVM (86.7%, best overall)
  │   │
  │   └── No ──► Have domain expert time? ──Yes──► Use Rules (66.6%)
  │       │
  │       └── No ──► Need zero-shot? ──Yes──► Use LLM (78.7%)
  │           │
  │           └── No ──► Can collect data? ──Yes──► Collect 100+ labels → SVM
  │               │
  │               └── No ──► Use Rules
  │
  └──► Need multi-label nuance? ──Yes──► Consider LLM for categories (94.9%)
      │
      └── No ──► SVM sufficient
```

### 6.2 Context-Specific Recommendations

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| **Production system, 100+ labels** | SVM or FastText | Best accuracy, lowest cost, proven reliability |
| **Rapid prototype, no data** | Zero-shot LLM | Immediate results, no data collection |
| **Regulatory (explainability)** | Rules or SVM | Interpretable decisions, auditable |
| **Resource-constrained (mobile/edge)** | FastText | Smallest model, CPU-only, fast |
| **High-velocity domain (changing vocab)** | SVM | Automatic adaptation via retraining |
| **Stable domain, need entities** | Entity-Augmented | If 4.8pp gap to SVM acceptable for interpretability |

---

## 7. Generalization Analysis

### 7.1 Cross-Domain Transfer

| Approach | Transferability | Why |
|----------|-----------------|-----|
| Rules | **Poor** | Domain-specific patterns |
| FastText | Moderate | Subword embeddings transfer partially |
| SVM | Moderate | TF-IDF vocabulary domain-specific |
| Entity-Augmented | **Poor** | Entity vocab highly domain-specific |
| LLM Zero-shot | **Excellent** | Pre-trained on broad corpus |
| GEPA | Unknown | Depends on prompt generalization |

### 7.2 Within-Domain Robustness

| Approach | Robustness to Drift | Mechanism |
|----------|----------------------|-----------|
| Rules | Poor | Manual updates required |
| FastText | Good | Retrain on new data |
| SVM | Good | Retrain on new data |
| Entity-Augmented | Poor | Vocabulary updates required |
| LLM Zero-shot | **Excellent** | No retraining needed |
| GEPA | Moderate | Prompt may generalize |

**Trade-off**: LLMs generalize across domains but underperform on specific tasks. Classical ML excels within domain but requires task-specific data.

---

## 8. Engineering Recommendations

### 8.1 For New Projects

**Phase 1: Baseline (Day 1)**
```bash
# Start with rules if zero data
01_handwritten_rules/classify.py

# Or zero-shot LLM for immediate prototype
06_llm/compare_base.py --test-size 20
```

**Phase 2: Production (Week 1-2)**
```bash
# Collect 100-200 labels, train SVM
03_classical_ml/train_eval.py

# Benchmark against rules/LLM baseline
07_evaluation/compare_all.py
```

**Phase 3: Optimization (Month 1+)**
```bash
# If LLM needed for specific sub-tasks (categories)
06_llm/gepa_optimise.py

# Or FastText for speed-critical deployment
02_fasttext/train.py
```

### 8.2 Migration Path

| From | To | When |
|------|-----|------|
| Rules | SVM | >50 labeled examples collected |
| SVM | FastText | Need faster inference, slight accuracy drop OK |
| SVM | Entity-Augmented | Need entity interpretability, 4.8pp drop acceptable |
| LLM Zero-shot | SVM | Labeled data available, cost reduction needed |
| LLM Zero-shot | GEPA | Need better accuracy, can afford optimization |

### 8.3 Anti-Patterns

❌ **Don't use spaCy textcat on small data** — 73.1% vs. 86.7% SVM

❌ **Don't use Entity-Augmented for rapidly changing domains** — vocabulary maintenance burden

❌ **Don't use LLM at scale without cost analysis** — 4700× latency, GPU costs

❌ **Don't skip SVM baseline** — it's the accuracy champion with lowest cost

---

## 9. Summary: The Engineering Hierarchy

### By Total Cost of Ownership (1M queries, 3-year horizon)

| Rank | Approach | TCO | Notes |
|------|----------|-----|-------|
| 🥇 1 | **SVM** | **Lowest** | Best accuracy, proven reliability |
| 🥈 2 | FastText | Low | Slightly lower accuracy, fastest |
| 🥉 3 | Rules | **High** | Maintenance burden |
| 4 | Entity-Augmented | Medium | Upfront + maintenance effort |
| 5 | LLM Zero-shot | **Very High** | Inference costs dominate |
| 6 | GEPA | High | Optimization + inference costs |

### By Risk (Operational + Maintenance)

| Rank | Approach | Risk Level | Notes |
|------|----------|-----------|-------|
| 🥇 1 | **SVM** | **Lowest** | Mature, stable, well-understood |
| 🥈 2 | FastText | Low | Simple, few failure modes |
| 🥉 3 | LLM Zero-shot | Moderate | API dependencies, latency variance |
| 4 | GEPA | Moderate | Optimization may fail |
| 5 | spaCy textcat | High | Neural brittleness |
| 6 | Entity-Augmented | **High** | Knowledge drift, maintenance |
| 7 | Rules | **Very High** | Expert dependency, decay |

---

## 10. Final Verdict

**For the facility support classification task**:

| Criterion | Winner |
|-----------|--------|
| **Accuracy** | SVM (86.7%) |
| **Speed** | FastText (1-3ms) |
| **Cost** | SVM / FastText (free, CPU-only) |
| **Maintainability** | SVM (scikit-learn stability) |
| **Zero-shot** | LLM (78.7%) — but below SVM |
| **Interpretability** | SVM (feature weights) or Rules |

**Default recommendation**: **SVM with TF-IDF** — state-of-the-art accuracy, lowest total cost, proven reliability.

**LLM role**: Use for rapid prototyping without data, or when GEPA proves it can close the 8pp gap to SVM.

---

*Last updated: 2026-06-01*
*GEPA results pending — update section 2.7 when available*
