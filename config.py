"""Project configuration.

Paths are derived from this file so the app does not depend on a
machine-specific Downloads folder. Label encoding matches the original
script: Fake = 0, Real = 1.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "samples"
ORIGINAL_DIR = PROJECT_ROOT / "original"

FAKE_CSV = DATA_DIR / "Fake.csv"
TRUE_CSV = DATA_DIR / "True.csv"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LR_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
NB_MODEL_PATH = MODELS_DIR / "naive_bayes.joblib"
SVM_MODEL_PATH = MODELS_DIR / "linear_svm.joblib"
METADATA_PATH = MODELS_DIR / "training_metadata.json"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
LEAKAGE_PATH = OUTPUTS_DIR / "leakage_report.json"

# Original encoding from fake_news_detection.py
LABEL_FAKE = 0
LABEL_REAL = 1
LABEL_NAMES = {LABEL_FAKE: "Fake-like", LABEL_REAL: "Real-like"}
POSITIVE_CLASS_ORIGINAL = LABEL_REAL  # original metrics used class 1 = real
POSITIVE_CLASS_DETECTION = LABEL_FAKE  # fake-news detection view

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Original TfidfVectorizer(stop_words='english', max_df=0.7)
TFIDF_PARAMS = {
    "stop_words": "english",
    "max_df": 0.7,
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 1,
}

LR_PARAMS = {"max_iter": 1000, "solver": "lbfgs"}
NB_PARAMS = {"alpha": 1.0}
SVM_PARAMS = {"max_iter": 2000, "dual": "auto"}

HYBRID_THRESHOLD = 0.5
MAX_INPUT_CHARS = 40_000
MAX_UPLOAD_BYTES = 1_000_000
TOP_N_FEATURES = 15

NEWSAPI_ENV_VAR = "NEWSAPI_KEY"
