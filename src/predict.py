"""Prediction helpers that preserve the original hybrid averaging rule."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin

from config import HYBRID_THRESHOLD, LABEL_FAKE, LABEL_NAMES, LABEL_REAL


def probability_for_label(model: ClassifierMixin, features, label: int) -> np.ndarray:
    """Return P(label) using the model's ``classes_`` alignment."""
    classes = list(model.classes_)
    if label not in classes:
        raise ValueError(f"Label {label} is not in model.classes_={classes}")
    index = classes.index(label)
    return np.asarray(model.predict_proba(features)[:, index], dtype=float)


def hybrid_real_probability(lr_real: float, nb_real: float) -> float:
    """Average of Logistic Regression and Naive Bayes P(real).

    Original formula from fake_news_detection.py::

        hybrid_prob = (lr_prob + nb_prob) / 2

    where ``lr_prob`` and ``nb_prob`` are ``predict_proba(...)[0][1]``,
    i.e. the probability of class 1 (real news).
    """
    return float((float(lr_real) + float(nb_real)) / 2.0)


def label_from_real_probability(real_probability: float, threshold: float = HYBRID_THRESHOLD) -> int:
    """Map P(real) to the original decision rule: real if probability >= 0.5."""
    return LABEL_REAL if real_probability >= threshold else LABEL_FAKE


@dataclass
class ModelScores:
    name: str
    real_probability: float
    fake_probability: float
    predicted_label: int

    @property
    def predicted_name(self) -> str:
        return LABEL_NAMES[self.predicted_label]


@dataclass
class PredictionResult:
    predicted_label: int
    hybrid_real_probability: float
    hybrid_fake_probability: float
    logistic_regression: ModelScores
    naive_bayes: ModelScores
    models_agree: bool
    word_count: int
    char_count: int
    threshold: float = HYBRID_THRESHOLD

    @property
    def predicted_name(self) -> str:
        return LABEL_NAMES[self.predicted_label]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["predicted_name"] = self.predicted_name
        return payload


def interpret_prediction(result: PredictionResult) -> str:
    """Short, careful wording for the UI. Scores are not factual guarantees."""
    agreement = (
        "Both models agree on the predicted class."
        if result.models_agree
        else "The two models disagree; the hybrid average is a compromise, not a tie-breaker of truth."
    )
    return (
        f"The hybrid ensemble estimates a {result.hybrid_real_probability:.1%} "
        f"model score for the Real-like class (label {LABEL_REAL}) and "
        f"{result.hybrid_fake_probability:.1%} for the Fake-like class "
        f"(label {LABEL_FAKE}). Predicted class: {result.predicted_name}. {agreement} "
        "These are uncalibrated model scores that describe resemblance to the "
        "training corpus, not independent verification of the article's claims."
    )


def predict_document(vectorizer, lr_model, nb_model, text: str) -> PredictionResult:
    """Transform one document with the saved vectorizer and score all models."""
    features = vectorizer.transform([text])
    lr_real = float(probability_for_label(lr_model, features, LABEL_REAL)[0])
    nb_real = float(probability_for_label(nb_model, features, LABEL_REAL)[0])
    hybrid_real = hybrid_real_probability(lr_real, nb_real)

    lr_label = label_from_real_probability(lr_real)
    nb_label = label_from_real_probability(nb_real)
    hybrid_label = label_from_real_probability(hybrid_real)

    lr_scores = ModelScores(
        name="Logistic Regression",
        real_probability=lr_real,
        fake_probability=1.0 - lr_real,
        predicted_label=lr_label,
    )
    nb_scores = ModelScores(
        name="Multinomial Naive Bayes",
        real_probability=nb_real,
        fake_probability=1.0 - nb_real,
        predicted_label=nb_label,
    )
    tokens = text.split()
    return PredictionResult(
        predicted_label=hybrid_label,
        hybrid_real_probability=hybrid_real,
        hybrid_fake_probability=1.0 - hybrid_real,
        logistic_regression=lr_scores,
        naive_bayes=nb_scores,
        models_agree=lr_label == nb_label,
        word_count=len(tokens),
        char_count=len(text),
    )
