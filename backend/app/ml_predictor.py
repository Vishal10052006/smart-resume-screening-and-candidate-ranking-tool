"""Supervised ML inference for the running API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from ml.features import build_pair_text_from_strings

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "resume_match_ridge.joblib"
_MODEL: Any = None


def load_model() -> Any:
    """Load the trained model once per process when the artifact exists."""
    global _MODEL
    if _MODEL is None and MODEL_PATH.exists():
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def reset_model_cache() -> None:
    """Clear the cached model; useful for tests and development reloads."""
    global _MODEL
    _MODEL = None


def predict_match_score(resume_text: str, job_description: str) -> float | None:
    """Predict a 0-100 match score; return None when no model is available."""
    model = load_model()
    if model is None:
        return None

    # Use the same feature construction as training to prevent train/inference drift.
    pair_text = build_pair_text_from_strings(resume_text, job_description)
    prediction = float(model.predict([pair_text])[0])
    return round(max(0.0, min(1.0, prediction)) * 100, 1)
