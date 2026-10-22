from src.bootstrap import ensure_project_root

ensure_project_root()

import pandas as pd
import streamlit as st

from src.ui import apply_theme, contribution_chart, init_state, require_models

apply_theme()
init_state()

st.title("Explainable AI")
st.caption(
    "Instance-level contributions from the fitted TF-IDF features. "
    "These do not prove that a story is true or false."
)

if require_models() is None:
    st.stop()

explanation = st.session_state.get("last_explanation")
text = st.session_state.get("last_text") or ""

if not explanation or not text:
    st.info("Analyse an article on the News Analysis page first. Explanations are computed for that article, not for the whole dataset.")
    st.stop()

if not explanation.get("available"):
    st.warning(explanation.get("reason") or "Explanation is not available for this article.")
    st.stop()

st.markdown(explanation["limitations"])

tabs = st.tabs(["Logistic Regression", "Naive Bayes"])

for tab, key in zip(tabs, ("logistic_regression", "naive_bayes")):
    with tab:
        block = explanation.get(key) or {}
        if not block.get("available"):
            st.warning(block.get("reason") or "Could not compute this explanation.")
            continue
        st.caption(block["method"])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                contribution_chart(block["toward_real"], "Pushes toward Real-like", "#1f6f4a"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                contribution_chart(block["toward_fake"], "Pushes toward Fake-like", "#9b2335"),
                use_container_width=True,
            )
        table = pd.DataFrame(block["toward_real"] + block["toward_fake"])
        table = table.sort_values("contribution", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("How to read this")
st.write(
    "- **Logistic Regression:** contribution = article TF-IDF weight × coefficient for class 1 (real). "
    "Positive values support Real-like; negative values support Fake-like."
)
st.write(
    "- **Naive Bayes:** contribution = article TF-IDF weight × (log P(feature|real) − log P(feature|fake))."
)
st.write(
    "A word such as `reuters` can dominate because True.csv articles often include a wire dateline, "
    "not because the model checked the claim."
)
