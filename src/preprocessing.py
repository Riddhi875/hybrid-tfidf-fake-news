"""Shared text preprocessing for training and inference.

The original project concatenated title and body, then passed the result
to TfidfVectorizer (which lowercases and applies English stop words).
This module keeps that behaviour so saved models stay aligned with the
training-time pipeline.
"""

from __future__ import annotations

import re

REUTERS_DATELINE = re.compile(
    r"^[A-Z][A-Z\s\./,'\-()]{0,80}\(Reuters\)\s*-\s*",
    flags=re.IGNORECASE,
)
WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip ends."""
    if text is None:
        return ""
    return WHITESPACE.sub(" ", str(text)).strip()


def combine_title_and_text(title: str | None, text: str | None) -> str:
    """Concatenate headline and body, matching the original script.

    Original: ``title.fillna('') + ' ' + text.fillna('')``
    Headline and body are combined because both were used at training time.
    Inference should do the same when a separate headline is provided.
    """
    return normalize_whitespace(f"{title or ''} {text or ''}")


def strip_reuters_dateline(text: str) -> str:
    """Remove a leading CITY (Reuters) - prefix when present.

    Used only in the robustness experiment. The primary model is trained
    on the original combined text so it matches the existing project.
    """
    cleaned = normalize_whitespace(text)
    return REUTERS_DATELINE.sub("", cleaned, count=1).strip()


def prepare_document(
    title: str | None = None,
    body: str | None = None,
    *,
    strip_dateline: bool = False,
) -> str:
    """Build the document string used by TF-IDF."""
    content = combine_title_and_text(title, body)
    if strip_dateline:
        content = strip_reuters_dateline(content)
    return content


def validate_input_text(text: str, max_chars: int) -> tuple[bool, str]:
    """Return (ok, message) for user-provided article text."""
    if text is None or not str(text).strip():
        return False, "Please provide an article, headline, or uploaded text file."
    cleaned = str(text).strip()
    if len(cleaned) > max_chars:
        return (
            False,
            f"Text is too long ({len(cleaned):,} characters). "
            f"Please shorten it to at most {max_chars:,} characters.",
        )
    if len(cleaned.split()) < 2 and len(cleaned) < 8:
        return False, "Input is too short to analyse. Paste a headline or a few sentences."
    return True, ""
