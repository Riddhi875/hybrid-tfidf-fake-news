from config import LABEL_FAKE, LABEL_REAL
from src.predict import hybrid_real_probability, label_from_real_probability


def test_hybrid_is_simple_average_of_real_probabilities():
    assert hybrid_real_probability(0.8, 0.4) == 0.6
    assert hybrid_real_probability(1.0, 0.0) == 0.5
    assert hybrid_real_probability(0.2, 0.2) == 0.2


def test_threshold_matches_original_cli():
    assert label_from_real_probability(0.5) == LABEL_REAL
    assert label_from_real_probability(0.4999) == LABEL_FAKE
    assert LABEL_FAKE == 0
    assert LABEL_REAL == 1
