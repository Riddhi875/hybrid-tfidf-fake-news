"""Held-out evaluation for LR, NB, the hybrid ensemble, and optional SVM."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import LABEL_FAKE, LABEL_REAL
from src.predict import hybrid_real_probability, label_from_real_probability, probability_for_label


def _safe_roc_auc(y_true, scores) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def metrics_for_positive_class(y_true, y_pred, scores_for_class_one, positive_label: int) -> dict[str, Any]:
    """Compute metrics with an explicit positive class.

    ``scores_for_class_one`` is P(real) or a decision score aligned so that
    larger values favour label 1. ROC-AUC is always computed with label 1
    as the score's positive direction, then we also report binary precision,
    recall, and F1 for the requested ``positive_label``.
    """
    return {
        "positive_label": int(positive_label),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "roc_auc_label1_as_score_positive": _safe_roc_auc(y_true, scores_for_class_one),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def confusion(y_true, y_pred) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[LABEL_FAKE, LABEL_REAL])
    return {
        "labels": [LABEL_FAKE, LABEL_REAL],
        "matrix": matrix.tolist(),
        "axis": "rows = actual [fake=0, real=1], columns = predicted [fake=0, real=1]",
    }


def evaluate_probability_model(name: str, y_true, y_pred, real_probability) -> dict[str, Any]:
    return {
        "name": name,
        "metrics_positive_real": metrics_for_positive_class(y_true, y_pred, real_probability, LABEL_REAL),
        "metrics_positive_fake": metrics_for_positive_class(y_true, y_pred, real_probability, LABEL_FAKE),
        "confusion_matrix": confusion(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[LABEL_FAKE, LABEL_REAL],
            target_names=["fake (0)", "real (1)"],
            zero_division=0,
        ),
        "roc_curve": _roc_payload(y_true, real_probability),
        "score_kind": "estimated_probability_of_real_class",
    }


def _roc_payload(y_true, scores) -> dict[str, Any] | None:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=LABEL_REAL)
    return {
        "false_positive_rate": fpr.tolist(),
        "true_positive_rate": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "positive_label": LABEL_REAL,
    }


def evaluate_models(vectorizer, lr_model, nb_model, X_test, y_test, svm_model=None) -> dict[str, Any]:
    """Evaluate every model on the same untouched test features."""
    lr_real = probability_for_label(lr_model, X_test, LABEL_REAL)
    nb_real = probability_for_label(nb_model, X_test, LABEL_REAL)
    hybrid_real = np.array(
        [hybrid_real_probability(a, b) for a, b in zip(lr_real, nb_real)],
        dtype=float,
    )
    lr_pred = np.array([label_from_real_probability(p) for p in lr_real])
    nb_pred = np.array([label_from_real_probability(p) for p in nb_real])
    hybrid_pred = np.array([label_from_real_probability(p) for p in hybrid_real])

    # sklearn .predict can theoretically differ from a 0.5 threshold on
    # P(real) if class priors change; we report both but use the original
    # threshold rule as the project decision rule.
    results = {
        "logistic_regression": evaluate_probability_model(
            "Logistic Regression", y_test, lr_model.predict(X_test), lr_real
        ),
        "naive_bayes": evaluate_probability_model(
            "Multinomial Naive Bayes", y_test, nb_model.predict(X_test), nb_real
        ),
        "hybrid_ensemble": evaluate_probability_model(
            "Hybrid TF-IDF Ensemble", y_test, hybrid_pred, hybrid_real
        ),
        "decision_rule": {
            "hybrid": "predict real if (P_LR(real) + P_NB(real)) / 2 >= 0.5",
            "individual_threshold_used_in_ui": "same 0.5 rule on each model's P(real)",
            "sklearn_predict_used_in_lr_nb_tables": True,
        },
        "hybrid_threshold_predictions": evaluate_probability_model(
            "Hybrid TF-IDF Ensemble", y_test, hybrid_pred, hybrid_real
        ),
        "note": (
            "metrics_positive_real matches the original script, which used "
            "sklearn defaults (pos_label=1, i.e. real news). "
            "metrics_positive_fake treats fake news as the positive class."
        ),
    }

    # Keep sklearn predict metrics for LR/NB so they match typical reports.
    results["logistic_regression_thresholded"] = evaluate_probability_model(
        "Logistic Regression (0.5 on P(real))", y_test, lr_pred, lr_real
    )
    results["naive_bayes_thresholded"] = evaluate_probability_model(
        "Naive Bayes (0.5 on P(real))", y_test, nb_pred, nb_real
    )

    if svm_model is not None:
        svm_pred = svm_model.predict(X_test)
        if hasattr(svm_model, "decision_function"):
            scores = np.asarray(svm_model.decision_function(X_test), dtype=float)
            score_kind = "decision_function_not_probability"
        elif hasattr(svm_model, "predict_proba"):
            scores = probability_for_label(svm_model, X_test, LABEL_REAL)
            score_kind = "estimated_probability_of_real_class"
        else:
            scores = svm_pred.astype(float)
            score_kind = "predicted_label_only"
        svm_eval = evaluate_probability_model("Linear SVM", y_test, svm_pred, scores)
        svm_eval["score_kind"] = score_kind
        results["linear_svm"] = svm_eval

    return results


def compact_metric_table(evaluation: dict[str, Any], view: str = "real") -> list[dict[str, Any]]:
    """Flatten metrics for charts. view is 'real' or 'fake'."""
    key = "metrics_positive_real" if view == "real" else "metrics_positive_fake"
    rows = []
    for model_key in ("logistic_regression", "naive_bayes", "hybrid_ensemble", "linear_svm"):
        block = evaluation.get(model_key)
        if not block:
            continue
        metrics = block[key]
        rows.append(
            {
                "model": block["name"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc_label1_as_score_positive"],
                "positive_label": metrics["positive_label"],
            }
        )
    return rows
