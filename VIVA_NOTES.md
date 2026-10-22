# Viva notes — Hybrid TF-IDF Ensemble for Explainable Fake News Detection

Use these answers in a 5–10 minute demonstration. Quote measured numbers from `outputs/EXPERIMENT_REPORT.md`, not from memory.

## What problem does this solve?

People need a way to screen whether a news article *looks like* previously labelled fake or real reporting. The system is a credibility-analysis aid. It does not decide whether a claim is objectively true.

## Why is NLP used?

The input is unstructured English text. Natural language processing turns headlines and bodies into features a classifier can use.

## Why use TF-IDF?

TF-IDF scores a term higher when it is important in this article and uncommon across the training corpus. It is transparent, fast, and standard for linear text classifiers. Unlike deep embeddings, each feature is a readable word, which makes coefficient-based explanation possible.

## Why use these two models?

**Logistic Regression** learns a linear weight for each TF-IDF term and outputs class scores. **Multinomial Naive Bayes** models class-conditional word probabilities and works well on sparse counts or TF-IDF. They are complementary: one is a discriminative linear model, the other a generative probability model. Both were in the original project.

## How does the ensemble work?

It does **not** vote on labels. It averages the two models’ estimated probabilities of the **real** class (label 1):

`P_hybrid(real) = (P_LR(real) + P_NB(real)) / 2`

If that value is at least 0.5, the article is Real-like; otherwise Fake-like. This is the original CLI formula.

## What makes the project different from a basic fake-news classifier?

1. The hybrid formula is documented and tested.
2. Predictions are explained with article-specific TF-IDF × model weights.
3. Evaluation reports both real-as-positive (original script) and fake-as-positive views.
4. Leakage checks cover duplicate overlap, Reuters datelines, and subject/label shortcuts.
5. The interface separates ML scores from optional external search.
6. Training is a one-off script; the web app only loads saved models.

## How was model performance evaluated?

A stratified 80/20 split with seed 42. The vectorizer is fit on training documents only. Accuracy, precision, recall, F1, confusion matrices, and ROC-AUC are computed on the held-out test set. A robustness run strips Reuters datelines and retrains on the same split without replacing the saved production models.

## What are the risks of data leakage?

If duplicate articles appear in both train and test, accuracy is inflated. If the vectorizer is fit on all data, test information leaks into the vocabulary. This dataset also has **source shortcuts**: True.csv is mostly Reuters wire copy; Fake.csv is not. A model can detect `WASHINGTON (Reuters)` instead of misinformation.

## Why is high accuracy not enough?

In-domain accuracy on a dataset with stylistic shortcuts does not mean the system can fact-check unseen publishers, new events, or adversarial wording. Precision and recall for the class you care about can tell a different story than overall accuracy.

## What are the current limitations?

- Labels come from source collections, not claim-level fact checks.
- No publisher-held-out or time-held-out external test set is bundled.
- Probabilities are not calibrated.
- Explanations are linear feature contributions, not causal proof.
- Optional NewsAPI search requires a key and never verifies truth.

## How could the system be improved further?

- Grouped splits by publisher or by removing datelines in the primary model.
- Calibration (Platt scaling) if scores are shown to users as probabilities.
- Transformer models with the same leakage controls.
- Human-in-the-loop review and claim matching against fact-check databases.
- Independent evaluation on later news from unseen outlets.
