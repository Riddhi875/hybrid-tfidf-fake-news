"""Streamlit home page for the explainable fake news detector."""

from __future__ import annotations

import json

import streamlit as st

from config import FAKE_CSV, LABEL_FAKE, LABEL_REAL, METRICS_PATH, TRUE_CSV
from src.ui import apply_theme, init_state, require_models
from src.ui_theme import apply_theme
apply_theme()
init_state()


def _metrics_preview():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


st.markdown('<div class="hero-card">', unsafe_allow_html=True)
st.markdown('<div class="kicker"> </div>', unsafe_allow_html=True)
st.markdown("# Explainable AI-Based Fake News Detection")
st.write(
    "A Hybrid TF-IDF ensemble that combines Logistic Regression and "
    "Multinomial Naive Bayes, then explains which terms pushed a single "
    "article toward a Fake-like or Real-like score."
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer"><strong>Disclaimer.</strong> This tool estimates '
    "whether an article resembles patterns learned from its training data. "
    "It does not independently establish factual truth, verify sources, or "
    "replace journalistic fact-checking.</div>",
    unsafe_allow_html=True,
)

st.subheader("How the system works")
st.markdown(
    """
<div class="workflow">
  <span>Input article</span><em>→</em>
  <span>Text preprocessing</span><em>→</em>
  <span>TF-IDF</span><em>→</em>
  <span>LR + Naive Bayes</span><em>→</em>
  <span>Hybrid ensemble</span><em>→</em>
  <span>Prediction and explanation</span>
</div>
""",
    unsafe_allow_html=True,
)

st.write(
    "Headline and body are concatenated, matching the original training script. "
    "TF-IDF converts the text into sparse features. Logistic Regression and "
    "Multinomial Naive Bayes each produce an estimated probability of the "
    "**real** class (label 1). The hybrid score is the average of those two "
    "probabilities. If the average is at least 0.5 the article is labelled "
    "Real-like; otherwise Fake-like."
)

col1, col2, col3 = st.columns(3)
payload = _metrics_preview()
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Dataset**")
    st.write("ISOT-style `Fake.csv` + `True.csv`")
    st.write("Columns: `title`, `text`, `subject`, `date`")
    st.write(f"Label mapping: fake = {LABEL_FAKE}, real = {LABEL_REAL}")
    st.caption(f"Expected files: `{FAKE_CSV.name}`, `{TRUE_CSV.name}` in `data/`")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Models**")
    st.write("- TF-IDF vectorizer (`stop_words='english'`, `max_df=0.7`)")
    st.write("- Logistic Regression")
    st.write("- Multinomial Naive Bayes")
    st.write("- Hybrid average of P(real)")
    st.write("- Linear SVM benchmark (evaluation only)")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Hybrid formula**")
    st.latex(r"P_{\text{hybrid}}(\text{real}) = \frac{P_{\text{LR}}(\text{real}) + P_{\text{NB}}(\text{real})}{2}")
    st.caption("Preserved from the original CLI detector. Scores are not calibrated guarantees.")
    st.markdown("</div>", unsafe_allow_html=True)

if payload:
    stats = payload["metadata"]["cleaning_stats"]
    st.subheader("Training corpus snapshot")
    a, b, c, d = st.columns(4)
    a.metric("Usable records", f"{stats['usable_rows']:,}")
    b.metric("Fake examples", f"{stats['class_counts']['fake']:,}")
    c.metric("Real examples", f"{stats['class_counts']['real']:,}")
    d.metric("Held-out test rows", f"{payload['metadata']['test_rows']:,}")
    st.caption(
        "Figures come from the last local training run. Open **Model Performance** "
        "for measured accuracy, precision, recall, F1, and ROC-AUC."
    )
else:
    st.warning("No saved evaluation yet. Train once, then reopen this page.")
    st.code("py -3 -m src.train_models")

artifacts = require_models()
if artifacts:
    st.success("Saved vectorizer and classifiers loaded. The app does not retrain on prediction.")

st.subheader("Navigate")
st.write(
    "Use the sidebar: **News Analysis** to score an article, **Explainable AI** "
    "for feature contributions, **Model Performance** for measured metrics, "
    "**Dataset Explorer** for corpus statistics, **Analysis History** for this "
    "session, and **Source Investigation** for optional related-news search."
)
