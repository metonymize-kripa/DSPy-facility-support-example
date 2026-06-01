# Conceptual Framework: Facility Support Message Classification

*A first-principles analysis of the problem space, design space, and baseline human process — prior to any technical solution choices.*

---

## 1. Problem Statement

Given an unstructured text message sent to a facility support organization (ProCare Facility Solutions in this dataset), produce three structured outputs:

- **Urgency**: one of {low, medium, high}
- **Sentiment**: one of {positive, neutral, negative}
- **Categories**: a subset of ten predefined service-request labels

The dataset is the [Meta llama-prompt-ops facility-support-analyzer corpus](https://github.com/meta-llama/llama-prompt-ops/tree/main/use-cases/facility-support-analyzer): 200 labeled email messages, each annotated with the three outputs above.

---

## 2. Cardinality of the Output Space

The three outputs are formally independent, giving a joint space of:

```
|Urgency| × |Sentiment| × |Categories|
= 3 × 3 × 2^10
= 9,216 possible outputs
```

In practice this overstates the problem. Looking at the dataset:

- Most messages activate 1–3 categories. The empirically reachable set of category combinations is well under 100 distinct subsets.
- Urgency and sentiment are not uniformly distributed. High urgency correlates strongly with emergency repair categories; low urgency with general inquiries.

**The effective problem space is roughly two orders of magnitude smaller than the formal space.** This matters because it means a relatively small labeled dataset (200 examples) can provide meaningful coverage, and that simple models with strong priors can perform surprisingly well.

---

## 3. Decomposition: Three Sub-Problems With Different Characters

The task is presented as one problem but decomposes into three with distinct difficulty profiles and distinct minimum viable solutions.

### 3.1 Sentiment (3-class)

**Signal type**: Primarily lexical — explicit emotional markers, complaint language, expressions of satisfaction.

**The hard case**: Professional correspondence conventions mask sentiment. A polite, courteous complaint ("I have always appreciated your exceptional services, however I must raise a concern...") carries negative affect behind positive-register framing. The first sentence tests as positive; the full arc is negative. A bag-of-words model fails here systematically. A model that reads the full message arc does better.

**Minimum viable model**: Bag-of-words classifier with a domain-adapted sentiment lexicon handles the majority of cases. The professional-register masking is the only structurally hard case.

### 3.2 Urgency (3-class ordinal)

**Signal type**: Two-layer — explicit markers and implicit domain knowledge.

*Explicit*: "immediately", "ASAP", "emergency", "cannot wait", "safety hazard". Near-deterministic when present.

*Implicit*: "My HVAC is making a noise" is low urgency in October and potentially high urgency in July. "A small leak" is routine unless guests arrive tomorrow or the building has sensitive equipment. Urgency at this level requires world knowledge about building operations and temporal context that a pure lexical model cannot supply.

**Minimum viable model**: Keyword/rule classifier covers ~70% of cases. The implicit urgency cases require either labeled examples (few-shot) or world knowledge (LLM).

### 3.3 Categories (multi-label over 10 labels)

**Signal type**: Vocabulary-driven primary signal; intent-reading for boundary cases.

The 10 labels form a specific business taxonomy:

| Label | Primary signal |
|-------|----------------|
| routine_maintenance_requests | "maintenance", "plumbing", "HVAC check" without emergency language |
| customer_feedback_and_complaints | Dissatisfaction with prior service, requests for remedies |
| training_and_support_requests | "guide me", "how do I", "training" |
| quality_and_safety_concerns | Mold, hazards, substandard outcomes, safety language |
| sustainability_and_environmental_practices | "eco-friendly", "green", "environmental" |
| cleaning_services_scheduling | Scheduling, timing, calendar adjustments |
| specialized_cleaning_services | "deep clean", "carpet", "window washing", "mold remediation" |
| emergency_repair_services | "broken", "failed", "urgent repair", active system failure |
| facility_management_issues | Building-level management, rent, structural concerns |
| general_inquiries | No specific request; information-seeking before commitment |

**The hard cases**:

1. *Co-occurrence*: `customer_feedback_and_complaints` and `quality_and_safety_concerns` frequently appear together but not always. `general_inquiries` co-occurs with almost everything — it is a residual category, not an exclusive one.

2. *Intent-based disambiguation*: A request to reschedule cleaning is `cleaning_services_scheduling` if it is a routine logistics adjustment, but is `customer_feedback_and_complaints` if the rescheduling is a response to poor service. The keyword "reschedule" alone does not resolve this — the intent of the message does.

3. *Taxonomy opacity*: These are not natural-language categories any reader would infer cold. They are a proprietary business taxonomy. A model without explicit knowledge of this taxonomy must infer it from labeled examples alone.

**Minimum viable model**: A supervised multi-label classifier trained on the provided labels. The taxonomy knowledge is embedded in the training signal. Vocabulary is the primary discriminant; intent disambiguation requires richer representations.

---

## 4. How a Human Expert Solves This

A trained call-center triage agent processes these messages via a **cheap-first decision cascade**, not a holistic reading:

**Step 1 — Subject line scan** (1 second). "Urgent HVAC Repair Needed" immediately resolves urgency=high and categories=[emergency_repair_services]. Subject lines are high-signal, low-noise prefixes. This step resolves a large fraction of cases.

**Step 2 — Explicit urgency markers** (2 seconds). Scan for "immediately", "ASAP", "cannot wait", "dangerous", "safety hazard". Their presence is near-deterministic. Their absence does not imply low urgency.

**Step 3 — Service type identification** (5 seconds). Keyword lookup against the known taxonomy: "HVAC" / "air conditioning" → emergency_repair or routine_maintenance (defer resolution to step 4). "carpet" / "deep clean" / "window washing" → specialized_cleaning. "leak" / "plumbing" → routine_maintenance or emergency (defer). "eco-friendly" → sustainability_and_environmental_practices.

**Step 4 — Message arc reading** (10–30 seconds for ambiguous cases). Read opening, middle escalation, and closing. Opening complaints about prior service → customer_feedback_and_complaints. Multiple prior failed contacts mentioned → urgency upgrade. Polite inquiry tone throughout → general_inquiries likely in scope.

**Step 5 — Default rule application**. `general_inquiries` is assigned when no specific request or complaint is identified. `customer_feedback_and_complaints` is added whenever the sender expresses dissatisfaction about prior service (not just the current situation).

This cascade is efficient because it front-loads cheap lexical checks and defers expensive intent-reading to cases that actually need it. It is a rational strategy under time pressure with high volume.

---

## 5. Domain Knowledge Taxonomy

A human expert relies on three distinct and separable types of knowledge:

### Type 1: Taxonomic Knowledge
*Knowing the label definitions and their boundaries.*

This is explicit, teachable knowledge. A new agent receives a glossary. It is not intuited from general language understanding. Any model — human or machine — must have this knowledge provided either directly (label definitions in the prompt, or training data) or inferred from examples.

### Type 2: Service Domain Knowledge
*Knowing that HVAC failure is more urgent than a missed cleaning appointment. That mold is a health hazard. That outdoor temperature context affects urgency interpretation.*

This is world knowledge about building operations, maintenance priorities, and physical systems. A junior agent develops this over months on the job. An LLM has an approximation of this from pretraining. A supervised classifier trained only on this dataset does not — it can only learn the correlations present in 200 examples.

### Type 3: Pragmatic/Register Knowledge
*Knowing that "I have always been satisfied with your services, but..." signals a complaint. That "prompt response would be appreciated" is a polite closing, not an urgency signal. That "I am writing to inquire" signals information-seeking, not a problem report.*

This is knowledge about professional correspondence conventions. It is the hardest to acquire because it is implicit — it is not stated in any glossary. It is learned through exposure to large amounts of professional communication. A junior agent without this knowledge will mis-classify polite complaints as neutral or positive. A bag-of-words model will do the same.

**The distribution of knowledge requirements across sub-tasks:**

| Sub-task | Type 1 (Taxonomy) | Type 2 (Domain) | Type 3 (Register) |
|----------|:-----------------:|:---------------:|:-----------------:|
| Sentiment | low | low | **high** |
| Urgency | low | **high** | medium |
| Categories | **high** | medium | medium |

This table is the theoretical foundation for the empirical comparison. Models that lack Type 3 knowledge (FastText, simple classifiers) will underperform specifically on sentiment. Models that lack Type 2 (small LLMs with limited world knowledge) will underperform specifically on implicit urgency. DSPy/GEPA prompt optimization is a mechanism for making Type 2 and Type 3 knowledge explicit and reliably applied — which is why it should help more on sentiment and urgency than on categories.

---

## 6. The Underappreciated Baseline: Rule-Based Systems

A domain expert with 30 minutes could write a rule-based classifier that covers a substantial fraction of cases:

```
IF subject contains ("urgent" OR "emergency" OR "immediate") THEN urgency = high
IF body contains ("HVAC" OR "air conditioning") AND NOT explicit emergency THEN
    categories += routine_maintenance_requests
IF body contains ("HVAC" OR "air conditioning") AND explicit emergency THEN
    categories += emergency_repair_services
IF body contains ("carpet" OR "deep clean" OR "window wash") THEN
    categories += specialized_cleaning_services
IF no specific request AND body contains ("inquire" OR "information" OR "question") THEN
    categories += general_inquiries
...
```

This is not a straw man. It is what most production triage systems ran before ML, and what many still run today. Including this baseline makes the contribution concrete: the paper shows the full ladder from handwritten rules → supervised small models → optimized prompts on small LLMs → frontier LLMs. The argument is not "LLMs beat FastText" but "here is precisely where each approach gains and loses, and why."

---

## 7. The Central Research Question

Given this analysis, the paper's central question is:

> **For this specific, narrow, domain-constrained classification task, at what point on the model complexity ladder does additional investment — in model size, in prompt optimization, or in labeled data — stop paying for itself?**

The three sub-tasks provide three different answers because they require different knowledge types. That is the theoretical contribution: a knowledge-type decomposition of a multi-output classification task that predicts, from first principles, which approaches will fail on which sub-tasks and why.

The empirical contribution is measuring where those predictions hold.

---

## 8. Design Space Summary

| Approach | Type 1 | Type 2 | Type 3 | Needs labels | Cost/query |
|----------|:------:|:------:|:------:|:------------:|:----------:|
| Handwritten rules | given | implicit | none | none | ~0 |
| FastText | from training | from training | none | yes (many) | ~0 |
| spaCy TextCat | from training | from training | none | yes (many) | ~0 |
| Local LLM, zero-shot | from prompt | from pretraining | partial | none | low |
| Small API LLM, zero-shot | from prompt | from pretraining | good | none | low |
| Small API LLM + GEPA | from prompt | from pretraining | **optimized** | few (train set) | low |
| Frontier LLM, zero-shot | from prompt | strong | strong | none | high |

The design space is not a single dimension. It is a three-dimensional space over the three knowledge types, with cost and labeled-data requirements as additional axes. The thesis of the paper is that prompt optimization (GEPA) can substitute for model scale specifically on Type 3 knowledge, and that Type 2 knowledge gaps are better closed with labeled examples or domain-specific fine-tuning than with larger models.

---

*Document version: initial draft. Revise as empirical results come in.*
