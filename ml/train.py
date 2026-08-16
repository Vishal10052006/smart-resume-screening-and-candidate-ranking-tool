"""Train and evaluate the supervised resume matching model.

Usage:
    python -m ml.train --data data/raw/resume_data_for_ranking.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.preprocessing import build_pair_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "resume_data_for_ranking.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "resume_match_ridge.joblib"
METRICS_PATH = MODEL_DIR / "evaluation.json"


def train(data_path: Path) -> dict:
    """Train the model, evaluate it on a hold-out set, and save artifacts."""
    df = pd.read_csv(data_path)

    required = {"matched_score", "job_position_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Combine candidate and job information into one supervised example.
    X = df.apply(build_pair_text, axis=1)
    y = df["matched_score"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            ("model", Ridge(alpha=10.0)),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = np.clip(pipeline.predict(X_test), 0.0, 1.0)

    metrics = {
        "model": "TF-IDF + Ridge Regression",
        "dataset_rows": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "random_state": 42,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the resume ranking model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    train(args.data)
