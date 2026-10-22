from src.bootstrap import ensure_project_root

ensure_project_root()

import streamlit as st

from config import MAX_INPUT_CHARS, MAX_UPLOAD_BYTES, SAMPLES_DIR
from src.ui import apply_theme, init_state, render_prediction, require_models, run_analysis

apply_theme()
init_state()

st.title("News Analysis")
st.caption("Paste text, split headline and body, upload a .txt file, or load a sample. Models are not retrained here.")

artifacts = require_models()
if artifacts is None:
    st.stop()

samples = {
    "Sensational demo (often Fake-like in this corpus)": SAMPLES_DIR / "sample_fake_like.txt",
    "Wire-style demo (often Real-like in this corpus)": SAMPLES_DIR / "sample_real_like.txt",
}

if "draft_headline" not in st.session_state:
    st.session_state.draft_headline = ""
if "draft_body" not in st.session_state:
    st.session_state.draft_body = ""

left, right = st.columns([2, 1])
with right:
    choice = st.selectbox("Load a demonstration article", ["(none)"] + list(samples))
    if st.button("Load sample") and choice != "(none)":
        path = samples[choice]
        st.session_state.draft_headline = path.stem.replace("_", " ")
        st.session_state.draft_body = path.read_text(encoding="utf-8")
        st.rerun()
    uploaded = st.file_uploader("Upload a plain-text .txt file", type=None)
    if uploaded is not None:
        name = uploaded.name.lower()
        if not name.endswith(".txt"):
            st.error("Unsupported file type. Please upload a .txt file.")
        elif uploaded.size > MAX_UPLOAD_BYTES:
            st.error("File is too large. Please upload at most 1 MB.")
        else:
            try:
                text = uploaded.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                st.error("The file is not valid UTF-8 text.")
            else:
                st.session_state.draft_body = text
                st.session_state.draft_headline = uploaded.name
                st.success("File loaded into the editor.")
    if st.button("Clear input and reset"):
        st.session_state.draft_headline = ""
        st.session_state.draft_body = ""
        st.session_state.last_result = None
        st.session_state.last_text = ""
        st.session_state.last_headline = ""
        st.session_state.last_explanation = None
        st.rerun()

with left:
    headline = st.text_input("Headline (optional)", key="draft_headline")
    body = st.text_area(
        "Article text",
        key="draft_body",
        height=280,
        placeholder="Paste the full article here…",
        max_chars=MAX_INPUT_CHARS,
    )
    analyse = st.button("Analyse article", type="primary")

if analyse:
    with st.spinner("Scoring with the saved TF-IDF models…"):
        result, error = run_analysis(headline, body)
    if error:
        st.warning(error)
    elif result:
        render_prediction(result)
elif st.session_state.last_result is not None:
    st.subheader("Latest analysis")
    render_prediction(st.session_state.last_result)
else:
    st.info("No analysis yet. Paste an article or load a sample, then choose Analyse article.")
