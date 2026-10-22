from src.preprocessing import (
    combine_title_and_text,
    prepare_document,
    strip_reuters_dateline,
    validate_input_text,
)


def test_combine_title_and_text_matches_original_join():
    assert combine_title_and_text("Hello", "World") == "Hello World"
    assert combine_title_and_text(None, "Body") == "Body"
    assert combine_title_and_text("Title", None) == "Title"


def test_reuters_dateline_stripped_once():
    raw = "WASHINGTON (Reuters) - The Senate voted on Tuesday."
    assert strip_reuters_dateline(raw) == "The Senate voted on Tuesday."


def test_prepare_document_optional_dateline():
    title = "Budget fight"
    body = "WASHINGTON (Reuters) - Lawmakers return Wednesday."
    combined = prepare_document(title, body)
    assert combined.startswith("Budget fight WASHINGTON")
    stripped = prepare_document(title, body, strip_dateline=True)
    assert "(Reuters)" not in stripped


def test_validate_empty_and_long_text():
    ok, message = validate_input_text("   ", 100)
    assert ok is False
    assert "provide" in message.lower()
    ok, message = validate_input_text("word", 100)
    assert ok is False
    ok, message = validate_input_text("a" * 50, 10)
    assert ok is False
    assert "too long" in message.lower()
    ok, message = validate_input_text("This is a valid short article about policy.", 10_000)
    assert ok is True
