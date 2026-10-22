"""Instance-level explanations for the TF-IDF linear / NB pipeline.

Contributions use the article's own TF-IDF values. Dataset-wide top words
are not treated as an explanation of a single prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config import LABEL_FAKE, LABEL_REAL, TOP_N_FEATURES


LIMITATIONS = (
    "These highlights show which TF-IDF features pushed this article's "
    "model score toward Fake-like or Real-like. They do not prove that a "
    "claim is true or false. Words can be important because of publisher "
    "style (for example a Reuters dateline), topic, or dataset artefacts."
)


def _present_feature_weights(vectorizer, features) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (indices, tfidf_values, feature_names) for nonzero entries."""
    row = features.tocsr()
    indices = row.indices
    values = row.data.astype(float)
    names = np.asarray(vectorizer.get_feature_names_out())
    return indices, values, names


def logistic_regression_contributions(vectorizer, lr_model, features, top_n: int = TOP_N_FEATURES) -> dict[str, Any]:
    """Contribution_i = TF-IDF_i * coefficient_i for the real class.

    sklearn binary Logistic Regression stores ``coef_`` for class 1 (real)
    versus class 0 (fake). Positive contribution supports Real-like;
    negative contribution supports Fake-like.
    """
    if not hasattr(lr_model, "coef_"):
        return {"available": False, "reason": "Logistic Regression coefficients are unavailable."}

    indices, tfidf_values, names = _present_feature_weights(vectorizer, features)
    coef = np.asarray(lr_model.coef_).ravel()
    contributions = tfidf_values * coef[indices]
    order = np.argsort(contributions)

    toward_fake = order[:top_n]
    toward_real = order[-top_n:][::-1]

    def pack(idx_subset):
        rows = []
        for local in idx_subset:
            rows.append(
                {
                    "feature": str(names[indices[local]]),
                    "tfidf": float(tfidf_values[local]),
                    "coefficient": float(coef[indices[local]]),
                    "contribution": float(contributions[local]),
                }
            )
        return rows

    return {
        "available": True,
        "method": "TF-IDF value × Logistic Regression coefficient for class 1 (real)",
        "toward_real": pack(toward_real),
        "toward_fake": pack(toward_fake),
        "intercept": float(lr_model.intercept_[0]) if hasattr(lr_model, "intercept_") else None,
        "limitations": LIMITATIONS,
        "class_encoding": {"fake": LABEL_FAKE, "real": LABEL_REAL},
    }


def naive_bayes_contributions(vectorizer, nb_model, features, top_n: int = TOP_N_FEATURES) -> dict[str, Any]:
    """Contribution_i = TF-IDF_i * (log P(f_i|real) - log P(f_i|fake)).

    This is a valid feature-log-probability contrast for MultinomialNB.
    It is not a SHAP value and is not a factual explanation of the news.
    """
    if not hasattr(nb_model, "feature_log_prob_"):
        return {"available": False, "reason": "Naive Bayes feature log-probabilities are unavailable."}

    classes = list(nb_model.classes_)
    try:
        real_idx = classes.index(LABEL_REAL)
        fake_idx = classes.index(LABEL_FAKE)
    except ValueError:
        return {"available": False, "reason": "Could not align Naive Bayes classes with labels 0/1."}

    indices, tfidf_values, names = _present_feature_weights(vectorizer, features)
    log_odds = nb_model.feature_log_prob_[real_idx] - nb_model.feature_log_prob_[fake_idx]
    contributions = tfidf_values * log_odds[indices]
    order = np.argsort(contributions)
    toward_fake = order[:top_n]
    toward_real = order[-top_n:][::-1]

    def pack(idx_subset):
        rows = []
        for local in idx_subset:
            rows.append(
                {
                    "feature": str(names[indices[local]]),
                    "tfidf": float(tfidf_values[local]),
                    "log_prob_real_minus_fake": float(log_odds[indices[local]]),
                    "contribution": float(contributions[local]),
                }
            )
        return rows

    return {
        "available": True,
        "method": "TF-IDF value × (feature_log_prob[real] − feature_log_prob[fake])",
        "toward_real": pack(toward_real),
        "toward_fake": pack(toward_fake),
        "limitations": LIMITATIONS,
        "class_encoding": {"fake": LABEL_FAKE, "real": LABEL_REAL},
    }


def explain_document(vectorizer, lr_model, nb_model, text: str, top_n: int = TOP_N_FEATURES) -> dict[str, Any]:
    if not text or not text.strip():
        return {"available": False, "reason": "No text was provided for explanation."}
    features = vectorizer.transform([text])
    if features.nnz == 0:
        return {
            "available": False,
            "reason": (
                "None of the tokens in this article appear in the training-time "
                "TF-IDF vocabulary, so feature contributions cannot be computed."
            ),
        }
    return {
        "available": True,
        "logistic_regression": logistic_regression_contributions(vectorizer, lr_model, features, top_n),
        "naive_bayes": naive_bayes_contributions(vectorizer, nb_model, features, top_n),
        "limitations": LIMITATIONS,
    }
