from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from config import LABEL_FAKE, LABEL_REAL
from src.explainability import explain_document
from src.predict import hybrid_real_probability, predict_document, probability_for_label
from src.train_models import load_trained_artifacts


def _tiny_models():
    texts = [
        "reuters washington senate vote budget",
        "reuters london diplomats signed treaty",
        "shocking secret you will not believe celebrity",
        "unbelievable viral hoax outrageous video",
    ]
    labels = [LABEL_REAL, LABEL_REAL, LABEL_FAKE, LABEL_FAKE]
    vectorizer = TfidfVectorizer(stop_words="english", max_df=1.0)
    features = vectorizer.fit_transform(texts)
    lr_model = LogisticRegression(max_iter=200).fit(features, labels)
    nb_model = MultinomialNB().fit(features, labels)
    return vectorizer, lr_model, nb_model


def test_probability_uses_class_alignment():
    vectorizer, lr_model, _ = _tiny_models()
    features = vectorizer.transform(["reuters washington senate"])
    real_p = probability_for_label(lr_model, features, LABEL_REAL)[0]
    fake_p = probability_for_label(lr_model, features, LABEL_FAKE)[0]
    assert abs(real_p + fake_p - 1.0) < 1e-9
    assert 0.0 <= real_p <= 1.0


def test_predict_document_hybrid_matches_average():
    vectorizer, lr_model, nb_model = _tiny_models()
    text = "reuters diplomats signed treaty in washington"
    result = predict_document(vectorizer, lr_model, nb_model, text)
    expected = hybrid_real_probability(
        result.logistic_regression.real_probability,
        result.naive_bayes.real_probability,
    )
    assert abs(result.hybrid_real_probability - expected) < 1e-12
    assert result.word_count > 0
    assert result.predicted_label in (LABEL_FAKE, LABEL_REAL)


def test_explanation_uses_article_features():
    vectorizer, lr_model, nb_model = _tiny_models()
    explanation = explain_document(vectorizer, lr_model, nb_model, "reuters senate budget vote")
    assert explanation["available"] is True
    lr_block = explanation["logistic_regression"]
    assert lr_block["available"] is True
    features = {row["feature"] for row in lr_block["toward_real"] + lr_block["toward_fake"]}
    assert "reuters" in features or "senate" in features or "budget" in features


def test_missing_models_message(tmp_path, monkeypatch):
    import src.train_models as train_models

    monkeypatch.setattr(train_models, "VECTORIZER_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(train_models, "LR_MODEL_PATH", tmp_path / "lr.joblib")
    monkeypatch.setattr(train_models, "NB_MODEL_PATH", tmp_path / "nb.joblib")
    try:
        load_trained_artifacts()
        raised = False
    except FileNotFoundError as exc:
        raised = True
        assert "py -3 -m src.train_models" in str(exc)
    assert raised is True
