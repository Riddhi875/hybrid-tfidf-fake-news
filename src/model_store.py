"""Load trained artifacts once for the Streamlit process."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.train_models import load_trained_artifacts


def get_artifacts() -> dict[str, Any]:
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _cached() -> dict[str, Any]:
            return load_trained_artifacts()

        return _cached()
    except ImportError:
        return _uncached_artifacts()


@lru_cache(maxsize=1)
def _uncached_artifacts() -> dict[str, Any]:
    return load_trained_artifacts()


def models_are_present() -> tuple[bool, str]:
    try:
        get_artifacts()
        return True, ""
    except FileNotFoundError as exc:
        return False, str(exc)
