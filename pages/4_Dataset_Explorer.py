from src.bootstrap import ensure_project_root

ensure_project_root()

import pandas as pd
import plotly.express as px
import streamlit as st

from config import LABEL_FAKE, LABEL_REAL
from src.dataset_analysis import class_term_tables, dataset_overview, length_histograms
from src.ui import apply_theme, init_state

apply_theme()
init_state()

st.title("Dataset Explorer")
st.caption(
    "Statistics are computed from local `data/Fake.csv` and `data/True.csv`. "
    "Visual preprocessing does not overwrite those files."
)


@st.cache_data(show_spinner="Loading dataset statistics…")
def _load_overview():
    overview = dataset_overview()
    frame = overview.pop("frame")
    return overview, frame


try:
    overview, frame = _load_overview()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

counts = overview["class_counts"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Usable records", f"{overview['usable_rows']:,}")
c2.metric("Fake (label 0)", f"{counts['fake']:,}")
c3.metric("Real (label 1)", f"{counts['real']:,}")
c4.metric("Fake share", f"{overview['class_balance']['fake']:.1%}")

st.write(
    f"Raw rows: Fake.csv = {overview['raw_rows']['fake']:,}, True.csv = {overview['raw_rows']['true']:,}. "
    f"Empty combined content dropped, then exact duplicate `title+text` rows dropped. "
    f"Duplicate content rows removed: {overview['cleaning_stats']['duplicate_content_rows']:,}."
)

missing_fake = overview["cleaning_stats"]["fake_missing"]
missing_true = overview["cleaning_stats"]["true_missing"]
st.write("Missing values in source files:", {"Fake.csv": missing_fake, "True.csv": missing_true})

st.warning(
    f"Reuters-style dateline share — fake: {overview['reuters_share']['fake']:.1%}, "
    f"real: {overview['reuters_share']['real']:.1%}. Frequent words can reflect publishers "
    "and collection method rather than truthfulness."
)

label_filter = st.multiselect(
    "Filter by label",
    options=[LABEL_FAKE, LABEL_REAL],
    default=[LABEL_FAKE, LABEL_REAL],
    format_func=lambda x: "Fake-like (0)" if x == LABEL_FAKE else "Real-like (1)",
)
max_len = int(frame["content_len"].quantile(0.99))
length_range = st.slider("Article word-count range", 0, max(max_len, 1), (0, max_len))
filtered = frame[frame["label"].isin(label_filter)]
filtered = filtered[filtered["content_len"].between(length_range[0], length_range[1])]

hist = length_histograms(filtered)
col_a, col_b = st.columns(2)
with col_a:
    fig = px.histogram(
        hist,
        x="content_len",
        color=hist["label"].map({LABEL_FAKE: "Fake", LABEL_REAL: "Real"}),
        barmode="overlay",
        nbins=40,
        title="Article length (words)",
        labels={"x": "Words", "color": "Class"},
    )
    st.plotly_chart(fig, use_container_width=True)
with col_b:
    fig = px.histogram(
        hist,
        x="title_len",
        color=hist["label"].map({LABEL_FAKE: "Fake", LABEL_REAL: "Real"}),
        barmode="overlay",
        nbins=30,
        title="Headline length (words)",
        labels={"color": "Class"},
    )
    st.plotly_chart(fig, use_container_width=True)

subject = pd.DataFrame(overview["subject_counts"])
subject["class"] = subject["label"].map({LABEL_FAKE: "Fake", LABEL_REAL: "Real"})
st.plotly_chart(
    px.bar(subject, x="subject", y="count", color="class", title="Subject counts by class"),
    use_container_width=True,
)

st.subheader("Common terms")
st.caption("Corpus frequencies on a sample. Not an explanation of any one prediction.")

@st.cache_data(show_spinner="Counting n-grams…")
def _terms(records):
    sample_frame = pd.DataFrame.from_records(records)
    return class_term_tables(sample_frame)


term_payload = _terms(filtered[["label", "content"]].to_dict(orient="records"))
t1, t2 = st.columns(2)
with t1:
    st.markdown("**Fake unigrams**")
    st.dataframe(pd.DataFrame(term_payload["fake"]["unigrams"], columns=["term", "count"]), hide_index=True)
    st.markdown("**Fake bigrams**")
    st.dataframe(pd.DataFrame(term_payload["fake"]["bigrams"], columns=["term", "count"]), hide_index=True)
with t2:
    st.markdown("**Real unigrams**")
    st.dataframe(pd.DataFrame(term_payload["real"]["unigrams"], columns=["term", "count"]), hide_index=True)
    st.markdown("**Real bigrams**")
    st.dataframe(pd.DataFrame(term_payload["real"]["bigrams"], columns=["term", "count"]), hide_index=True)

try:
    from wordcloud import WordCloud
except ImportError:
    st.caption("Word clouds skipped because the optional `wordcloud` package is not installed.")
else:
    import matplotlib.pyplot as plt

    st.subheader("Word clouds (optional)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name, terms in (
        (axes[0], "Fake", term_payload["fake"]["unigrams"]),
        (axes[1], "Real", term_payload["real"]["unigrams"]),
    ):
        if not terms:
            ax.set_title(f"{name} (no terms)")
            ax.axis("off")
            continue
        cloud = WordCloud(width=500, height=300, background_color="white").generate_from_frequencies(dict(terms))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(name)
    st.pyplot(fig)
