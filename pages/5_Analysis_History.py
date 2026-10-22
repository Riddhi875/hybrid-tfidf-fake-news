from src.bootstrap import ensure_project_root

ensure_project_root()

import pandas as pd
import streamlit as st

from src.ui import apply_theme, init_state
from src.ui_theme import apply_theme
apply_theme()
init_state()

st.title("Analysis History")
st.caption(
    "Session-only history stored in memory. Full article text is not saved. "
    "Nothing is sent to a third-party service from this page."
)

history = st.session_state.get("history") or []
if not history:
    st.info("No analyses in this session yet.")
else:
    frame = pd.DataFrame(history)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect a row", options=list(range(len(frame))), format_func=lambda i: f"{frame.loc[i, 'timestamp']} — {frame.loc[i, 'headline']}")
    st.json(history[selected])

if st.button("Clear session history"):
    st.session_state.history = []
    st.rerun()
