from src.bootstrap import ensure_project_root

ensure_project_root()

import streamlit as st

from src.evidence_search import build_search_links, fetch_newsapi_headlines, newsapi_configured
from src.ui import apply_theme, init_state
from src.ui_theme import apply_theme
apply_theme()
init_state()

st.title("News Evidence and Source Investigation")
st.warning(
    "This page is separate from the classifier. Search results never change the "
    "machine-learning prediction. Related articles can repeat the same claim, "
    "so agreement among headlines is not verification."
)

query = st.text_input(
    "Headline or key phrase",
    value=st.session_state.get("last_headline") or "",
)
st.caption("The core app works without an API key. External search links always work.")

if query.strip():
    st.subheader("Open a search in your browser")
    for link in build_search_links(query):
        st.markdown(f"- [{link['provider']}]({link['url']})")

    st.subheader("Optional NewsAPI integration")
    if not newsapi_configured():
        st.info(
            "NEWSAPI_KEY is not configured. Set that environment variable to enable "
            "headline retrieval. No sources are invented while it is missing."
        )
    if st.button("Fetch NewsAPI headlines"):
        with st.spinner("Contacting NewsAPI…"):
            payload = fetch_newsapi_headlines(query)
        st.write(payload["message"])
        if payload["articles"]:
            for article in payload["articles"]:
                title = article["title"] or "(untitled)"
                url = article["url"]
                source = article["source"] or "Unknown publisher"
                date = article["published_at"] or "date not supplied"
                if url:
                    st.markdown(f"- [{title}]({url}) — {source}, {date}")
                    st.caption(article.get("description") or "")
                else:
                    st.markdown(f"- {title} — {source}, {date} (no URL supplied)")
        elif newsapi_configured():
            st.write("No useful evidence was returned.")
else:
    st.info("Enter a headline or phrase to generate search links.")

if st.session_state.get("last_result") is not None:
    result = st.session_state.last_result
    st.divider()
    st.caption(
        f"Current ML prediction remains {result.predicted_name} "
        f"(hybrid Real-like score {result.hybrid_real_probability:.1%}). "
        "This page does not override it."
    )
