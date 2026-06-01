# 03 — Classical ML

TF-IDF vectorisation + scikit-learn classifiers. The standard supervised
baseline between FastText and neural/LLM approaches.

## Motivation

Classical ML (TF-IDF + Logistic Regression, Naive Bayes, SVM) is interpretable,
fast to train, and has well-understood failure modes. On small datasets (200 examples)
it often outperforms FastText due to better regularisation and feature selection.
It also produces feature weights that explain *why* it made a prediction —
useful for validating the rule-based classifier's intuitions.

## Approaches

| Model | Notes |
|-------|-------|
| TF-IDF + Logistic Regression | Strong general baseline; L2 regularisation |
| TF-IDF + Naive Bayes (Multinomial) | Interpretable; poor on correlated features |
| TF-IDF + Linear SVM | Often best on small text datasets |
| TF-IDF + Random Forest | Non-linear; good for feature importance analysis |

All use `sklearn.multioutput.MultiOutputClassifier` for the categories sub-task.

## Scripts

| Script | Purpose |
|--------|---------|
| `train_eval.py` | Train all models with cross-validation, report per-task accuracy |
| `feature_analysis.py` | Top features per label — what words drive each category? |

## Run

```bash
uv run --group classical 03_classical_ml/train_eval.py
```

## Expected outcome

~68–78% aggregate. LogReg and SVM likely outperform FastText on this small dataset.
Feature analysis provides interpretability that bridges the rules approach and
the LLM approaches.
