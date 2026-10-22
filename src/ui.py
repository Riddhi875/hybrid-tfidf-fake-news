"""Shared Streamlit helpers and visual theme."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import LABEL_FAKE, LABEL_NAMES, LABEL_REAL
from src.bootstrap import ensure_project_root
from src.explainability import explain_document
from src.model_store import get_artifacts, models_are_present
from src.predict import PredictionResult, interpret_prediction, predict_document
from src.preprocessing import prepare_document, validate_input_text
from config import MAX_INPUT_CHARS

ensure_project_root()

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: "Source Sans 3", "Segoe UI", sans-serif;
}
.stApp {
  background: #f4f1ea;
  color: #1b2430;
}
section[data-testid="stSidebar"] {
  background: #15202b;
}
section[data-testid="stSidebar"] * {
  color: #e8eef4 !important;
}
.hero-card, .metric-card, .panel-card {
  background: #fffdf8;
  border: 1px solid #ddd4c4;
  border-radius: 16px;
  padding: 1.25rem 1.4rem;
  box-shadow: 0 8px 24px rgba(21, 32, 43, 0.06);
  margin-bottom: 1rem;
}
.hero-card h1 {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 2rem;
  color: #15202b;
  margin-bottom: 0.35rem;
}
.kicker {
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 0.75rem;
  color: #8a3b2d;
  font-weight: 700;
}
.disclaimer {
  background: #fff4e8;
  border-left: 4px solid #b4532a;
  padding: 0.9rem 1rem;
  border-radius: 8px;
  color: #4a3428;
}
.pred-fake {
  background: #fdecee;
  color: #8a1f2b;
  border: 1px solid #f3c4ca;
  border-radius: 12px;
  padding: 1rem 1.1rem;
}
.pred-real {
  background: #e8f6ee;
  color: #1c5c3a;
  border: 1px solid #bfe3cd;
  border-radius: 12px;
  padding: 1rem 1.1rem;
}
.workflow {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
}
.workflow span {
  background: #15202b;
  color: #f4f1ea;
  padding: 0.45rem 0.7rem;
  border-radius: 999px;
  font-size: 0.85rem;
}
.workflow em {
  color: #8a3b2d;
  font-style: normal;
  font-weight: 700;
}
footer { visibility: hidden; }
</style>
"""


def apply_theme() -> None:
    st.set_page_config(
        page_title="Explainable Fake News Detection",
        page_icon="📰",
        layout="wide",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def require_models() -> dict[str, Any] | None:
    present, message = models_are_present()
    if not present:
        st.error("Trained models are missing.")
        st.code("py -3 -m src.train_models")
        st.caption(message)
        return None
    return get_artifacts()


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_text", "")
    st.session_state.setdefault("last_headline", "")
    st.session_state.setdefault("last_explanation", None)


def record_history(headline: str, result: PredictionResult) -> None:
    st.session_state.history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "headline": (headline or "").strip()[:180] or "(no headline)",
            "predicted_class": result.predicted_name,
            "hybrid_real": round(result.hybrid_real_probability, 4),
            "hybrid_fake": round(result.hybrid_fake_probability, 4),
            "lr_real": round(result.logistic_regression.real_probability, 4),
            "nb_real": round(result.naive_bayes.real_probability, 4),
            "agreement": "Agree" if result.models_agree else "Disagree",
        },
    )


def run_analysis(headline: str, body: str) -> tuple[PredictionResult | None, str]:
    document = prepare_document(headline, body)
    ok, message = validate_input_text(document, MAX_INPUT_CHARS)
    if not ok:
        return None, message
    artifacts = get_artifacts()
    result = predict_document(
        artifacts["vectorizer"],
        artifacts["logistic_regression"],
        artifacts["naive_bayes"],
        document,
    )
    explanation = explain_document(
        artifacts["vectorizer"],
        artifacts["logistic_regression"],
        artifacts["naive_bayes"],
        document,
    )
    st.session_state.last_result = result
    st.session_state.last_text = document
    st.session_state.last_headline = headline
    st.session_state.last_explanation = explanation
    record_history(headline, result)
    return result, ""


def score_bar(result: PredictionResult) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=["Logistic Regression", "Naive Bayes", "Hybrid ensemble"],
        y=[
            result.logistic_regression.real_probability,
            result.naive_bayes.real_probability,
            result.hybrid_real_probability,
        ],
        name="Real-like score",
        marker_color="#1f6f4a",
    )
    fig.add_bar(
        x=["Logistic Regression", "Naive Bayes", "Hybrid ensemble"],
        y=[
            result.logistic_regression.fake_probability,
            result.naive_bayes.fake_probability,
            result.hybrid_fake_probability,
        ],
        name="Fake-like score",
        marker_color="#9b2335",
    )
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1], title="Estimated model score"),
        legend=dict(orientation="h"),
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_prediction(result: PredictionResult) -> None:
    css_class = "pred-real" if result.predicted_label == LABEL_REAL else "pred-fake"
    st.markdown(
        f'<div class="{css_class}"><strong>Predicted class:</strong> {result.predicted_name}'
        f"<br>Hybrid Real-like score: {result.hybrid_real_probability:.1%}"
        f" &nbsp;|&nbsp; Fake-like score: {result.hybrid_fake_probability:.1%}"
        f"<br>Models {'agree' if result.models_agree else 'disagree'}.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(score_bar(result), use_container_width=True)
    with right:
        st.markdown("**Individual model outputs**")
        st.write(
            pd.DataFrame(
                [
                    {
                        "Model": "Logistic Regression",
                        "Class": result.logistic_regression.predicted_name,
                        "Real-like score": result.logistic_regression.real_probability,
                        "Fake-like score": result.logistic_regression.fake_probability,
                    },
                    {
                        "Model": "Multinomial Naive Bayes",
                        "Class": result.naive_bayes.predicted_name,
                        "Real-like score": result.naive_bayes.real_probability,
                        "Fake-like score": result.naive_bayes.fake_probability,
                    },
                    {
                        "Model": "Hybrid ensemble",
                        "Class": result.predicted_name,
                        "Real-like score": result.hybrid_real_probability,
                        "Fake-like score": result.hybrid_fake_probability,
                    },
                ]
            )
        )
        st.caption(
            "These values are estimated class scores from `predict_proba`, not calibrated "
            "probabilities of factual correctness."
        )
        st.info(interpret_prediction(result))
        st.write(
            f"Words: {result.word_count:,} · Characters: {result.char_count:,} · "
            f"Decision rule: Real-like if hybrid P(real) ≥ 0.5 "
            f"(labels: fake={LABEL_FAKE}, real={LABEL_REAL})."
        )


def contribution_chart(rows: list[dict[str, Any]], title: str, color: str) -> go.Figure:
    ordered = list(reversed(rows))
    fig = go.Figure(
        go.Bar(
            x=[row["contribution"] for row in ordered],
            y=[row["feature"] for row in ordered],
            orientation="h",
            marker_color=color,
        )
    )
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Contribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
