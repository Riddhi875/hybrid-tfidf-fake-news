"""Load and clean the ISOT-style Fake.csv / True.csv pair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import FAKE_CSV, LABEL_FAKE, LABEL_REAL, RANDOM_SEED, TEST_SIZE, TRUE_CSV
from src.preprocessing import combine_title_and_text, strip_reuters_dateline


REQUIRED_COLUMNS = ("title", "text")


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place Fake.csv and True.csv in the data/ directory. "
            "These files are the original Kaggle/ISOT-style tables with "
            "columns title, text, subject, date."
        )
    return path


def load_raw_frames(
    fake_path: Path | None = None,
    true_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two source CSVs without changing labels."""
    fake_path = _require_file(Path(fake_path or FAKE_CSV))
    true_path = _require_file(Path(true_path or TRUE_CSV))
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    for name, frame in (("Fake.csv", fake_df), ("True.csv", true_df)):
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
    return fake_df, true_df


def _annotate(frame: pd.DataFrame, label: int, source_file: str) -> pd.DataFrame:
    out = frame.copy()
    for column in ("title", "text", "subject", "date"):
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)
    out["label"] = int(label)
    out["source_file"] = source_file
    out["content"] = [
        combine_title_and_text(title, text)
        for title, text in zip(out["title"], out["text"])
    ]
    out["content_no_dateline"] = out["content"].map(strip_reuters_dateline)
    out["has_reuters_dateline"] = out["text"].str.contains(r"\(Reuters\)", regex=True)
    return out


def load_labelled_dataset(
    fake_path: Path | None = None,
    true_path: Path | None = None,
    *,
    drop_empty: bool = True,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a labelled frame plus cleaning statistics.

    Label mapping (from the original script, not assumed):
    - Fake.csv rows receive label 0 (fake)
    - True.csv rows receive label 1 (real)
    """
    fake_df, true_df = load_raw_frames(fake_path, true_path)
    stats: dict[str, Any] = {
        "fake_raw_rows": int(len(fake_df)),
        "true_raw_rows": int(len(true_df)),
        "fake_missing": fake_df.isna().sum().to_dict(),
        "true_missing": true_df.isna().sum().to_dict(),
        "fake_exact_duplicate_rows": int(fake_df.duplicated().sum()),
        "true_exact_duplicate_rows": int(true_df.duplicated().sum()),
        "label_mapping": {"fake": LABEL_FAKE, "real": LABEL_REAL},
        "columns": {
            "fake": list(fake_df.columns),
            "true": list(true_df.columns),
        },
    }

    labelled = pd.concat(
        [
            _annotate(fake_df, LABEL_FAKE, "Fake.csv"),
            _annotate(true_df, LABEL_REAL, "True.csv"),
        ],
        ignore_index=True,
    )
    stats["combined_rows_before_cleaning"] = int(len(labelled))
    stats["empty_content_before"] = int((labelled["content"].str.strip() == "").sum())
    stats["reuters_share_by_label"] = {
        str(LABEL_FAKE): float(labelled.loc[labelled["label"] == LABEL_FAKE, "has_reuters_dateline"].mean()),
        str(LABEL_REAL): float(labelled.loc[labelled["label"] == LABEL_REAL, "has_reuters_dateline"].mean()),
    }

    if drop_empty:
        labelled = labelled[labelled["content"].str.strip() != ""].copy()
    empty_body = labelled["text"].str.strip() == ""
    stats["rows_with_empty_body_kept_via_title"] = int(empty_body.sum())

    duplicate_mask = labelled.duplicated(subset=["content"], keep="first")
    stats["duplicate_content_rows"] = int(duplicate_mask.sum())
    if drop_duplicates:
        labelled = labelled.loc[~duplicate_mask].copy()

    labelled = labelled.reset_index(drop=True)
    stats["usable_rows"] = int(len(labelled))
    stats["class_counts"] = {
        "fake": int((labelled["label"] == LABEL_FAKE).sum()),
        "real": int((labelled["label"] == LABEL_REAL).sum()),
    }
    if "subject" in labelled.columns:
        subject_counts = (
            labelled.groupby(["source_file", "subject"]).size().reset_index(name="count")
        )
        stats["subjects"] = subject_counts.to_dict(orient="records")
    else:
        stats["subjects"] = []
    return labelled, stats


def stratified_split(frame: pd.DataFrame, content_column: str = "content"):
    """Stratified train/test split. Vectorizer must be fit on train only."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=frame["label"],
    )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        content_column,
    )


def leakage_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    """Exact content overlap between splits and subject/label association."""
    train_set = set(train_df["content"])
    test_set = set(test_df["content"])
    overlap = train_set.intersection(test_set)
    report: dict[str, Any] = {
        "exact_content_overlap": int(len(overlap)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "note": (
            "Exact duplicate content was dropped before splitting when "
            "drop_duplicates=True. Remaining overlap should be zero."
        ),
    }
    if "subject" in train_df.columns:
        subject_label = (
            pd.concat([train_df, test_df], ignore_index=True)
            .groupby("subject")["label"]
            .agg(["mean", "count"])
            .reset_index()
        )
        report["subject_mean_label"] = subject_label.to_dict(orient="records")
        report["subject_shortcut_warning"] = (
            "Real articles in this dataset use subjects politicsNews and "
            "worldnews; fake articles use a disjoint subject set. Subject "
            "is therefore a near-perfect source shortcut and is not used "
            "as a model feature."
        )
    reuters_train = float(train_df["has_reuters_dateline"].mean()) if "has_reuters_dateline" in train_df else None
    reuters_test = float(test_df["has_reuters_dateline"].mean()) if "has_reuters_dateline" in test_df else None
    report["reuters_dateline_share"] = {"train": reuters_train, "test": reuters_test}
    report["reuters_warning"] = (
        "A large majority of True.csv articles contain a Reuters dateline "
        "such as 'WASHINGTON (Reuters) -', while Fake.csv almost never does. "
        "A bag-of-words model can exploit this publishing artefact."
    )
    return report
