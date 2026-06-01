# 05 — Cobweb / Conceptual Clustering

Cobweb is an incremental concept-formation system (Fisher 1987) that builds
a classification tree by maximising Category Utility — a measure of how
much knowing the category helps predict attribute values. It requires no
labeled training data: it discovers structure in the data.

## Motivation

This is the most exploratory approach. The question it asks is:
**does the dataset have natural cluster structure that aligns with the task labels,
and can an unsupervised method recover it without any training signal?**

If Cobweb clusters correlate strongly with urgency or categories, that tells
us the task has strong latent structure that any reasonable model should capture.
If they don't, the task requires label-specific supervision.

## Approach

Two configurations:

### 5a — Raw text Cobweb
- TF-IDF vectorise messages → discretise to binary features (top-k terms present/absent)
- Run Cobweb on the binary feature matrix
- Evaluate: do recovered concepts align with urgency / category labels?

### 5b — Keyword-feature Cobweb
- Hand-defined binary features: presence of urgency keywords, service type keywords,
  sentiment markers, subject-line signals
- Run Cobweb on this interpretable feature space
- Advantage: the discovered concepts are human-readable

## Cobweb library

The [`cobweb`](https://github.com/cmaclell/concept_formation) Python package
by Christopher MacLellan implements Cobweb/3 with numeric and nominal attributes.

**Dataset size note**: Cobweb is designed for incremental learning on small-to-medium
datasets — 200 examples is well within its intended range. This is where it has
an advantage over deep learning approaches.

## Scripts

| Script | Purpose |
|--------|---------|
| `cobweb_text.py` | TF-IDF → binary features → Cobweb, cluster-label alignment |
| `cobweb_keywords.py` | Interpretable keyword features → Cobweb |
| `visualize_tree.py` | Print / plot the Cobweb concept tree |

## Run

```bash
uv run --group cobweb 05_cobweb_dm/cobweb_text.py
uv run --group cobweb 05_cobweb_dm/cobweb_keywords.py
```

## Research question

Does Cobweb recover urgency levels as natural concepts from raw text?
Does it distinguish emergency repair from routine maintenance without labels?
Results go into the paper as evidence for (or against) natural problem structure.

## References

- Fisher, D. H. (1987). Knowledge acquisition via incremental conceptual clustering. *Machine Learning*, 2(2), 139–172.
- MacLellan, C. J. et al. (2016). Trestle: Incremental learning in structured domains. *AAAI*.
