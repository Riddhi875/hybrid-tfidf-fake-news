# Hybrid TF-IDF Ensemble for Explainable Fake News Detection and News Credibility Analysis

College minor project. The web application scores an article as **Fake-like** or **Real-like** using a saved TF-IDF pipeline, shows each model's score, and explains which terms pushed the decision.

The classifier estimates resemblance to its training corpus. It does **not** independently establish factual truth.

## Problem statement

Online readers are flooded with political and celebrity content that mixes reporting, opinion, and fabricated stories. A lightweight, explainable screening tool can support media-literacy demonstrations and coursework. Deep black-box detectors are hard to defend in a viva; a TF-IDF linear ensemble is inspectable.

## Objectives

- Preserve the original label encoding, TF-IDF settings, models, and hybrid average.
- Separate training from inference so the app never retrains on open.
- Report measured test metrics instead of hardcoded accuracy.
- Explain individual predictions from coefficients and feature log-probabilities.
- Document leakage risks in the ISOT-style Fake/True collection.

## Features

- Home overview, workflow, disclaimer, and dataset snapshot
- News analysis: paste text, optional headline, `.txt` upload, samples, reset
- Per-model scores, hybrid score, agreement/disagreement, text statistics
- Explainable AI charts for Logistic Regression and Naive Bayes
- Model comparison: accuracy, precision, recall, F1, confusion matrices, ROC
- Dataset explorer with class balance, lengths, subjects, n-grams
- Session analysis history (no full article text stored)
- Optional source investigation via search links or NewsAPI

## System architecture

```
Input article
    → combine title + body (same as original script)
    → TfidfVectorizer loaded from training
    → Logistic Regression  ↘
    → Multinomial Naive Bayes → P_hybrid(real) = (P_LR(real) + P_NB(real)) / 2
    → threshold 0.5 → Fake-like (0) or Real-like (1)
    → instance-level feature contributions
```

| Path | Role |
| --- | --- |
| `app.py` | Streamlit home page |
| `pages/` | Analysis, XAI, metrics, dataset, history, sources |
| `src/train_models.py` | Train, evaluate, save artifacts |
| `src/predict.py` | Inference and hybrid formula |
| `src/explainability.py` | Instance contributions |
| `original/fake_news_detection.py` | Unmodified original CLI |

## Technology stack

Python 3.12, pandas, scikit-learn, joblib, Streamlit, Plotly, Matplotlib, seaborn, pytest.

## Dataset description and source

The original project used Kaggle/ISOT-style files `Fake.csv` and `True.csv` (Ahmed, Traore, Saad; commonly distributed as “Fake and real news dataset”). Copies live in `data/` (gitignored).

### Columns

`title`, `text`, `subject`, `date`

### Label mapping (from the original script — not assumed)

| File | Label | UI name |
| --- | --- | --- |
| `Fake.csv` | `0` | Fake-like |
| `True.csv` | `1` | Real-like |

Measured raw sizes when this repository was built: Fake 23,481 rows; True 21,417 rows. After dropping empty combined content and exact duplicate `title+text` strings, usable counts are written to `models/training_metadata.json`.

## Data preprocessing

1. Fill missing title/text with empty strings.
2. Combine `title + ' ' + text` (original behaviour).
3. Drop rows whose combined content is empty.
4. Drop exact duplicate content to reduce train/test leakage.
5. Stratified 80/20 split, `random_state=42`.
6. Fit TF-IDF **only** on the training split.

Inference uses the same concatenation and the saved vectorizer. It does not refit vocabulary.

## TF-IDF

`TfidfVectorizer(stop_words='english', max_df=0.7)` — same configuration as the original script. Default lowercase is enabled. Unigrams only, matching the original.

## Logistic Regression

Linear classifier on TF-IDF features, `max_iter=1000`. `predict_proba` column for class `1` is P(real). Coefficients × TF-IDF values explain a single article.

## Multinomial Naive Bayes

Class-conditional token model on the same sparse features. Explanation uses TF-IDF × (`feature_log_prob[real] − feature_log_prob[fake]`).

## Hybrid ensemble formula

Preserved from `original/fake_news_detection.py`:

```text
lr_prob = lr_model.predict_proba(input_tfidf)[0][1]   # P(real)
nb_prob = nb_model.predict_proba(input_tfidf)[0][1]   # P(real)
hybrid_prob = (lr_prob + nb_prob) / 2
if hybrid_prob >= 0.5: Real-like
else: Fake-like
```

The implementation looks up class columns through `model.classes_` so the average stays aligned if class order changes.

## Training instructions

```powershell
cd C:\Users\HP\hybrid-tfidf-fake-news
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -3 -m src.train_models
```

Training writes `models/*.joblib`, `models/training_metadata.json`, `outputs/metrics.json`, plots, and `outputs/EXPERIMENT_REPORT.md`.

Serialized models must be loaded only from this trusted training run.

## Application launch

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The app loads saved models. Opening a page does not retrain.

## Evaluation methodology and measured results

- Stratified hold-out, seed 42, vectorizer fit on train only.
- Metrics are computed twice: **positive class = real (1)** (original sklearn default used in the CLI) and **positive class = fake (0)** (fake-news detection view).
- Linear SVM is an optional benchmark. Its ROC uses a decision function, not a probability, unless noted in `metrics.json`.
- A robustness experiment retrains on Reuters-dateline-stripped text **without** replacing the saved app models.

After you train, copy the printed table from `outputs/EXPERIMENT_REPORT.md` into reports and slides. Do not invent accuracy numbers.

## Example usage

1. Train once.
2. Launch Streamlit.
3. Open **News Analysis**, load a sample, click **Analyse article**.
4. Open **Explainable AI** for contributing terms.
5. Open **Model Performance** and state which class is positive.
6. Open **Source Investigation** and show that search links do not change the prediction.

## Screenshots

Capture the Home, News Analysis, Explainable AI, and Model Performance pages during your demo and add them here if your report requires figures.

## Limitations

- High in-domain accuracy is expected because True.csv is mostly Reuters wire copy and Fake.csv is not.
- Subject names are disjoint across classes and are not used as features, but writing style still correlates with source.
- Scores are uncalibrated model outputs.
- No bundled publisher-held-out contemporary news set.
- Word clouds require optional `wordcloud`.

## Future scope

Publisher-grouped splits, probability calibration, transformer baselines with the same leakage protocol, claim matching against fact-check archives, and evaluation on later unseen outlets.

## Ethical considerations

Do not present a Fake-like score as a finding of guilt, defamation, or “debunking.” The tool can amplify dataset bias (politics, US-centric sources, 2016–2017 events). Do not upload private documents to NewsAPI unless you intend that data to leave the machine. Load only models you trained.

## Acknowledgements and licensing

Original CLI: `original/fake_news_detection.py`. Dataset: ISOT / Kaggle fake-and-real-news distribution; follow the licence of the copy you downloaded. scikit-learn and Streamlit are used under their respective licences.

## Commands (Windows)

```powershell
cd C:\Users\HP\hybrid-tfidf-fake-news
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -3 -m src.train_models
py -3 -m pytest -q
streamlit run app.py
```
