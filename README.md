# Spam Email Classifier

An interpretable logistic regression spam classifier built on a real-world email dataset, with full feature engineering, cross-validation, model analysis, and an ethical discussion of automated content filtering.

**[View Report →](https://hire-arvin.github.io/spam-classifier/spam_classifier.html)**

## What's Inside

- **Feature engineering**: word indicator features plus structural and lexical signals (HTML tags, capitalization rate, exclamation count, reply indicators, keyword flags) — recall rose from 11% to 60% when word indicators were replaced by these
- **Logistic regression** (scikit-learn) with 10-fold cross-validation: **86.5% validation accuracy** (87.4% mean CV) against a 74.5% majority-class baseline, at 85% precision and 60% recall
- **Model analysis**: ROC curve, feature coefficient visualization, feature ablation study showing which signals drive classifications
- **Ethics section**: ground truth ambiguity, the real cost of false positives, and why interpretability is a design requirement when automated decisions affect people

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib

## Files

| File | Description |
|------|-------------|
| `spam_classifier.qmd` | Quarto source — full write-up with embedded code |
| `spam_classifier.html` | Self-contained rendered report (open in browser) |
| `utils.py` | Feature extraction and data loading helpers |
