"""Dataset summaries for the explorer page. Does not mutate training CSVs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from config import LABEL_FAKE, LABEL_REAL
from src.data_loader import load_labelled_dataset, load_raw_frames


def dataset_overview() -> dict[str, Any]:
    labelled, stats = load_labelled_dataset(drop_empty=True, drop_duplicates=True)
    fake_raw, true_raw = load_raw_frames()
    labelled = labelled.copy()
    labelled["title_len"] = labelled["title"].str.split().str.len()
    labelled["text_len"] = labelled["text"].str.split().str.len()
    labelled["content_len"] = labelled["content"].str.split().str.len()
    labelled["char_len"] = labelled["content"].str.len()

    return {
        "cleaning_stats": stats,
        "raw_rows": {"fake": int(len(fake_raw)), "true": int(len(true_raw))},
        "usable_rows": int(len(labelled)),
        "class_counts": stats["class_counts"],
        "class_balance": {
            "fake": stats["class_counts"]["fake"] / stats["usable_rows"],
            "real": stats["class_counts"]["real"] / stats["usable_rows"],
        },
        "length_summary": labelled[["title_len", "text_len", "content_len", "char_len", "label"]]
        .groupby("label")
        .describe()
        .to_dict(),
        "subject_counts": labelled.groupby(["label", "subject"]).size().reset_index(name="count").to_dict(orient="records"),
        "reuters_share": {
            "fake": float(labelled.loc[labelled["label"] == LABEL_FAKE, "has_reuters_dateline"].mean()),
            "real": float(labelled.loc[labelled["label"] == LABEL_REAL, "has_reuters_dateline"].mean()),
        },
        "frame": labelled,
    }


def length_histograms(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["label", "title_len", "content_len", "char_len", "subject"]].copy()


def top_ngrams(texts: list[str], ngram_range=(1, 1), top_n: int = 20) -> list[tuple[str, int]]:
    if not texts:
        return []
    vectorizer = CountVectorizer(stop_words="english", ngram_range=ngram_range, max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    names = vectorizer.get_feature_names_out()
    order = np.argsort(frequencies)[::-1][:top_n]
    return [(str(names[i]), int(frequencies[i])) for i in order]


def class_term_tables(frame: pd.DataFrame, sample_per_class: int = 4000) -> dict[str, Any]:
    """Frequent terms per class on a sample. These are corpus stats, not explanations."""
    out: dict[str, Any] = {"note": "Frequent terms reflect topics and source style, not truthfulness."}
    for label, name in ((LABEL_FAKE, "fake"), (LABEL_REAL, "real")):
        subset = frame.loc[frame["label"] == label, "content"]
        sample = subset.sample(min(sample_per_class, len(subset)), random_state=42).tolist()
        out[name] = {
            "unigrams": top_ngrams(sample, (1, 1)),
            "bigrams": top_ngrams(sample, (2, 2)),
        }
    return out


def wordcloud_frequencies(terms: list[tuple[str, int]]) -> dict[str, int]:
    return {token: count for token, count in terms}
