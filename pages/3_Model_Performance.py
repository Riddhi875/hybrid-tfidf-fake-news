from src.bootstrap import ensure_project_root

ensure_project_root()

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import LABEL_FAKE, LABEL_REAL, METRICS_PATH
from src.ui import apply_theme, init_state, require_models
from src.ui_theme import apply_theme
apply_theme()
init_state()

st.title("Model Comparison and Performance")
st.caption("Every number on this page is read from the last training run's measured test-set evaluation.")

if require_models() is None:
    st.stop()

if not METRICS_PATH.exists():
    st.error("outputs/metrics.json was not found. Train the models first.")
    st.code("py -3 -m src.train_models")
    st.stop()

payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
view = st.radio(
    "Positive class for precision, recall, and F1",
    ("Real (label 1) — matches the original script", "Fake (label 0) — fake-news detection view"),
    horizontal=True,
)
table_key = "metric_table_positive_real" if view.startswith("Real") else "metric_table_positive_fake"
table = pd.DataFrame(payload[table_key])
st.dataframe(
    table.style.format(
        {col: "{:.4f}" for col in ("accuracy", "precision", "recall", "f1") if col in table}
    ),
    use_container_width=True,
    hide_index=True,
)

st.info(
    f"Fake news is label {LABEL_FAKE}; real news is label {LABEL_REAL}. "
    "ROC-AUC uses the continuous Real-like score (larger means more Real-like). "
    "Linear SVM uses a decision function, not a probability, when shown."
)

metric_long = table.melt(
    id_vars=["model"],
    value_vars=["accuracy", "precision", "recall", "f1"],
    var_name="metric",
    value_name="value",
)
fig = px.bar(
    metric_long,
    x="metric",
    y="value",
    color="model",
    barmode="group",
    range_y=[0, 1],
    title="Held-out test metrics",
)
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Confusion matrices")
evaluation = payload["evaluation"]
cols = st.columns(3 if not evaluation.get("linear_svm") else 4)
keys = ["logistic_regression", "naive_bayes", "hybrid_ensemble"]
if evaluation.get("linear_svm"):
    keys.append("linear_svm")
for col, key in zip(cols, keys):
    block = evaluation[key]
    matrix = np.array(block["confusion_matrix"]["matrix"])
    heat = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Fake (0)", "Real (1)"],
        y=["Fake (0)", "Real (1)"],
        title=block["name"],
    )
    heat.update_layout(coloraxis_showscale=False, height=360)
    col.plotly_chart(heat, use_container_width=True)

st.subheader("ROC curves")
roc_fig = go.Figure()
for key in keys:
    roc = evaluation[key].get("roc_curve")
    auc = evaluation[key]["metrics_positive_real"]["roc_auc_label1_as_score_positive"]
    if not roc:
        continue
    roc_fig.add_trace(
        go.Scatter(
            x=roc["false_positive_rate"],
            y=roc["true_positive_rate"],
            mode="lines",
            name=f"{evaluation[key]['name']} (AUC={auc:.3f})" if auc is not None else evaluation[key]["name"],
        )
    )
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="#444")))
roc_fig.update_layout(
    xaxis_title="False positive rate",
    yaxis_title="True positive rate",
    yaxis=dict(range=[0, 1]),
    xaxis=dict(range=[0, 1]),
    height=420,
)
st.plotly_chart(roc_fig, use_container_width=True)

st.subheader("What the metrics mean")
st.markdown(
    """
- **Accuracy** — share of test articles whose predicted label matches the dataset label.
- **Precision** — of the articles predicted as the chosen positive class, how many truly had that label.
- **Recall** — of the articles that truly belong to the positive class, how many were found.
- **F1-score** — harmonic mean of precision and recall.
- **ROC-AUC** — ranking quality of the continuous Real-like score; 0.5 is chance, 1.0 is perfect ranking.
"""
)

best = payload.get("best_model_by_macro_f1", {})
st.write(
    f"Highest ranked model by macro-F1 in the saved evaluation: **{best.get('name')}**. "
    f"Hybrid uniquely best: **{best.get('hybrid_is_best')}**."
)

if payload.get("robustness_no_dateline"):
    st.subheader("Robustness check: Reuters dateline stripped")
    st.write(payload["robustness_no_dateline"]["description"])
    st.dataframe(pd.DataFrame(payload["robustness_no_dateline"]["metric_table_positive_real"]), hide_index=True, use_container_width=True)

with st.expander("Classification reports"):
    for key in keys:
        st.text(evaluation[key]["classification_report"])
