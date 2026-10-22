from config import LABEL_FAKE, LABEL_NAMES, LABEL_REAL


def test_original_label_encoding_is_preserved():
    assert LABEL_FAKE == 0
    assert LABEL_REAL == 1
    assert LABEL_NAMES[0] == "Fake-like"
    assert LABEL_NAMES[1] == "Real-like"
