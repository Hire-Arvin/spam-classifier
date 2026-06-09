# Spam Email Classifier

An interpretable logistic regression spam classifier built on a real-world email dataset, with full feature engineering, cross-validation, model analysis, and an ethical discussion of automated content filtering.

**[View Report →](https://hire-arvin.github.io/spam-classifier/spam_classifier.html)**

## What's Inside

- **Feature engineering**: word indicator features plus structural and lexical signals (HTML ratio, punctuation density, message length)
- **Logistic regression** trained from scratch and cross-validated; achieves >95% accuracy at a balanced precision/recall tradeoff
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
