"""Train TF-IDF models, evaluate on a held-out test set, and save artifacts.

Fit the vectorizer on training documents only. The hybrid ensemble is not
refit; it averages P(real) from Logistic Regression and Naive Bayes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from config import (
    LABEL_FAKE,
    LABEL_REAL,
    LR_MODEL_PATH,
    LR_PARAMS,
    METADATA_PATH,
    METRICS_PATH,
    MODELS_DIR,
    NB_MODEL_PATH,
    NB_PARAMS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SVM_MODEL_PATH,
    SVM_PARAMS,
    TEST_SIZE,
    TFIDF_PARAMS,
    VECTORIZER_PATH,
    LEAKAGE_PATH,
)
from src.data_loader import leakage_checks, load_labelled_dataset, stratified_split
from src.evaluation import compact_metric_table, evaluate_models


def train_and_save(
    *,
    include_svm: bool = True,
    robustness_no_dateline: bool = True,
) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    frame, cleaning_stats = load_labelled_dataset()
    train_df, test_df, content_column = stratified_split(frame)
    leakage = leakage_checks(train_df, test_df)

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(train_df[content_column])
    X_test = vectorizer.transform(test_df[content_column])
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    lr_model = LogisticRegression(random_state=RANDOM_SEED, **LR_PARAMS)
    lr_model.fit(X_train, y_train)

    nb_model = MultinomialNB(**NB_PARAMS)
    nb_model.fit(X_train, y_train)

    svm_model = None
    if include_svm:
        svm_model = LinearSVC(random_state=RANDOM_SEED, **SVM_PARAMS)
        svm_model.fit(X_train, y_train)

    evaluation = evaluate_models(vectorizer, lr_model, nb_model, X_test, y_test, svm_model=svm_model)

    robustness = None
    if robustness_no_dateline:
        robustness = _robustness_without_dateline(train_df, test_df)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(lr_model, LR_MODEL_PATH)
    joblib.dump(nb_model, NB_MODEL_PATH)
    if svm_model is not None:
        joblib.dump(svm_model, SVM_MODEL_PATH)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "label_mapping": {"fake": LABEL_FAKE, "real": LABEL_REAL},
        "tfidf_params": TFIDF_PARAMS,
        "lr_params": LR_PARAMS,
        "nb_params": NB_PARAMS,
        "svm_params": SVM_PARAMS if include_svm else None,
        "hybrid_formula": "hybrid_P_real = (P_LR(real) + P_NB(real)) / 2",
        "hybrid_threshold": 0.5,
        "content_construction": "title + ' ' + text, matching the original script",
        "vectorizer_fit": "training split only",
        "cleaning_stats": cleaning_stats,
        "vocabulary_size": int(len(vectorizer.get_feature_names_out())),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "class_counts_train": {
            "fake": int((train_df["label"] == LABEL_FAKE).sum()),
            "real": int((train_df["label"] == LABEL_REAL).sum()),
        },
        "class_counts_test": {
            "fake": int((test_df["label"] == LABEL_FAKE).sum()),
            "real": int((test_df["label"] == LABEL_REAL).sum()),
        },
        "warning": "Load serialized models only from this trusted training run.",
    }

    payload = {
        "metadata": metadata,
        "evaluation": _jsonable(evaluation),
        "metric_table_positive_real": compact_metric_table(evaluation, "real"),
        "metric_table_positive_fake": compact_metric_table(evaluation, "fake"),
        "robustness_no_dateline": robustness,
        "best_model_by_macro_f1": _best_by_macro_f1(evaluation),
    }

    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LEAKAGE_PATH.write_text(json.dumps(leakage, indent=2, default=str), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    _write_plots(evaluation)
    _write_reports(payload, leakage)
    return payload


def _robustness_without_dateline(train_df, test_df) -> dict[str, Any]:
    """Retrain on text with Reuters datelines stripped; do not save as primary."""
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(train_df["content_no_dateline"])
    X_test = vectorizer.transform(test_df["content_no_dateline"])
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    lr_model = LogisticRegression(random_state=RANDOM_SEED, **LR_PARAMS).fit(X_train, y_train)
    nb_model = MultinomialNB(**NB_PARAMS).fit(X_train, y_train)
    evaluation = evaluate_models(vectorizer, lr_model, nb_model, X_test, y_test, svm_model=None)
    return {
        "description": (
            "Same split and models, but CITY (Reuters) - prefixes were stripped "
            "before TF-IDF. This checks how much the primary model may rely on "
            "a source artefact. These weights are not saved for the web app."
        ),
        "metric_table_positive_real": compact_metric_table(evaluation, "real"),
        "metric_table_positive_fake": compact_metric_table(evaluation, "fake"),
    }


def _best_by_macro_f1(evaluation: dict[str, Any]) -> dict[str, Any]:
    scored = []
    for key in ("logistic_regression", "naive_bayes", "hybrid_ensemble", "linear_svm"):
        block = evaluation.get(key)
        if not block:
            continue
        scored.append((block["metrics_positive_real"]["f1_macro"], block["name"]))
    scored.sort(reverse=True)
    winner_name = scored[0][1] if scored else None
    unique_scores = {round(s, 6) for s, _ in scored}
    return {
        "name": winner_name,
        "ranking_macro_f1_positive_real_table": scored,
        "hybrid_is_best": bool(winner_name == "Hybrid TF-IDF Ensemble") and len(unique_scores) > 1,
        "note": "Do not claim the hybrid is best unless this ranking says so.",
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _write_plots(evaluation: dict[str, Any]) -> None:
    sns.set_theme(style="whitegrid")
    models = ["logistic_regression", "naive_bayes", "hybrid_ensemble"]
    if evaluation.get("linear_svm"):
        models.append("linear_svm")

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    if len(models) == 1:
        axes = [axes]
    for ax, key in zip(axes, models):
        block = evaluation[key]
        matrix = np.array(block["confusion_matrix"]["matrix"])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(block["name"])
        ax.set_xlabel("Predicted (0=fake, 1=real)")
        ax.set_ylabel("Actual (0=fake, 1=real)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrices.png", dpi=140)
    plt.close(fig)

    plt.figure(figsize=(8, 6))
    for key in models:
        roc = evaluation[key].get("roc_curve")
        auc = evaluation[key]["metrics_positive_real"]["roc_auc_label1_as_score_positive"]
        if not roc or auc is None:
            continue
        plt.plot(
            roc["false_positive_rate"],
            roc["true_positive_rate"],
            label=f"{evaluation[key]['name']} (AUC={auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False positive rate (real as positive score)")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on the held-out test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "roc_curve.png", dpi=140)
    plt.close()


def _write_reports(payload: dict[str, Any], leakage: dict[str, Any]) -> None:
    lines = [
        "# Experiment report",
        "",
        "This report is generated by `src/train_models.py` from measured test-set results.",
        "Do not paste unverified accuracy numbers into documentation.",
        "",
        "## Configuration",
        f"- Random seed: {payload['metadata']['random_seed']}",
        f"- Test size: {payload['metadata']['test_size']}",
        f"- Label mapping: fake={LABEL_FAKE}, real={LABEL_REAL}",
        f"- Hybrid formula: {payload['metadata']['hybrid_formula']}",
        f"- Usable rows after cleaning: {payload['metadata']['cleaning_stats']['usable_rows']}",
        f"- Train rows: {payload['metadata']['train_rows']}",
        f"- Test rows: {payload['metadata']['test_rows']}",
        "",
        "## Leakage and shortcut checks",
        f"- Exact content overlap between train and test: {leakage.get('exact_content_overlap')}",
        f"- {leakage.get('reuters_warning')}",
        f"- {leakage.get('subject_shortcut_warning', '')}",
        "",
        "## Metrics with positive class = real (label 1)",
        "This view matches the original script's sklearn defaults.",
        "",
    ]
    for row in payload["metric_table_positive_real"]:
        lines.append(
            f"- {row['model']}: accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}, ROC-AUC={row['roc_auc']}"
        )
    lines += ["", "## Metrics with positive class = fake (label 0)", ""]
    for row in payload["metric_table_positive_fake"]:
        lines.append(
            f"- {row['model']}: accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}"
        )
    ranking = payload["best_model_by_macro_f1"]
    lines += [
        "",
        "## Ranking",
        f"- Highest macro-F1 (from the real-positive tables' macro F1 field in evaluation JSON): {ranking['name']}",
        f"- Hybrid is uniquely best: {ranking['hybrid_is_best']}",
        "",
        "## Robustness (Reuters dateline stripped)",
    ]
    if payload.get("robustness_no_dateline"):
        for row in payload["robustness_no_dateline"]["metric_table_positive_real"]:
            lines.append(
                f"- {row['model']}: accuracy={row['accuracy']:.4f}, f1={row['f1']:.4f}, ROC-AUC={row['roc_auc']}"
            )
    lines += [
        "",
        "## Limitations",
        "- The ISOT-style Fake/True split is known to mix source style with the label.",
        "- High in-domain accuracy does not imply real-world fact checking.",
        "- Independent publisher-held-out news was not available in this repository.",
        "",
    ]
    (OUTPUTS_DIR / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    report_txt = []
    evaluation = payload["evaluation"]
    for key in ("logistic_regression", "naive_bayes", "hybrid_ensemble", "linear_svm"):
        block = evaluation.get(key)
        if not block:
            continue
        report_txt.append(f"===== {block['name']} =====")
        report_txt.append(block["classification_report"])
        report_txt.append("")
    (OUTPUTS_DIR / "classification_reports.txt").write_text("\n".join(report_txt), encoding="utf-8")


def load_trained_artifacts() -> dict[str, Any]:
    """Load saved models. Does not train."""
    required = {
        "vectorizer": VECTORIZER_PATH,
        "logistic_regression": LR_MODEL_PATH,
        "naive_bayes": NB_MODEL_PATH,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Trained models were not found:\n- "
            + "\n- ".join(missing)
            + "\n\nFrom the project root run:\n  py -3 -m src.train_models"
        )
    artifacts = {name: joblib.load(path) for name, path in required.items()}
    artifacts["linear_svm"] = joblib.load(SVM_MODEL_PATH) if SVM_MODEL_PATH.exists() else None
    artifacts["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    artifacts["metrics"] = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    artifacts["leakage"] = json.loads(LEAKAGE_PATH.read_text(encoding="utf-8")) if LEAKAGE_PATH.exists() else {}
    return artifacts


def main() -> None:
    payload = train_and_save()
    print("Training complete.")
    print(f"Models saved under {MODELS_DIR}")
    print(f"Metrics saved to {METRICS_PATH}")
    print("Held-out metrics (positive class = real / label 1):")
    for row in payload["metric_table_positive_real"]:
        print(
            f"  {row['model']}: acc={row['accuracy']:.4f} "
            f"prec={row['precision']:.4f} rec={row['recall']:.4f} "
            f"f1={row['f1']:.4f} auc={row['roc_auc']}"
        )


if __name__ == "__main__":
    main()
