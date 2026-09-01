"""Train, validate, and save the supervised resume-ranking model.

Usage:
    python -m ml.train --data data/processed/cleaned_resume_data.csv

The evaluation reports both a standard random hold-out and a stricter
job-position-grouped hold-out. The final artifact is refit on all rows only
after evaluation is complete.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from ml.preprocessing import build_pair_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "cleaned_resume_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "resume_match_ridge.joblib"
METRICS_PATH = MODEL_DIR / "evaluation.json"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Create the reproducible TF-IDF + Ridge baseline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Calculate bounded regression metrics."""
    clipped = np.clip(predictions, 0.0, 1.0)
    return {
        "mae": round(float(mean_absolute_error(y_true, clipped)), 4),
        "rmse": round(float(mean_squared_error(y_true, clipped) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, clipped)), 4),
    }


def mean_ndcg_by_group(
    y_true: pd.Series,
    predictions: np.ndarray,
    groups: pd.Series,
    k: int = 10,
) -> float:
    """Measure ranking quality within each job-position group."""
    frame = pd.DataFrame(
        {"target": y_true.to_numpy(), "prediction": predictions, "group": groups.to_numpy()}
    )
    scores: list[float] = []
    for _, group in frame.groupby("group"):
        if len(group) < 2:
            continue
        limit = min(k, len(group))
        scores.append(
            float(
                ndcg_score(
                    group["target"].to_numpy().reshape(1, -1),
                    group["prediction"].to_numpy().reshape(1, -1),
                    k=limit,
                )
            )
        )
    return round(float(np.mean(scores)), 4) if scores else 0.0


def evaluate_split(
    X: pd.Series,
    y: pd.Series,
    train_index: np.ndarray,
    test_index: np.ndarray,
    groups: pd.Series,
) -> dict:
    """Fit a fresh model on one split and return regression/ranking metrics."""
    model = build_pipeline()
    model.fit(X.iloc[train_index], y.iloc[train_index])
    predictions = np.clip(model.predict(X.iloc[test_index]), 0.0, 1.0)

    metrics = regression_metrics(y.iloc[test_index], predictions)
    metrics["ndcg_at_10"] = mean_ndcg_by_group(
        y.iloc[test_index], predictions, groups.iloc[test_index], k=10
    )
    metrics["train_samples"] = int(len(train_index))
    metrics["test_samples"] = int(len(test_index))
    metrics["features"] = int(len(model.named_steps["tfidf"].vocabulary_))
    return metrics


def train(data_path: Path) -> dict:
    """Evaluate the model, refit on all data, and save the final artifact."""
    df = pd.read_csv(data_path)
    required = {"matched_score", "job_position_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    y = pd.to_numeric(df["matched_score"], errors="raise").astype(float)
    groups = df["job_position_name"].fillna("unknown").astype(str)
    X = df.apply(build_pair_text, axis=1)

    # Standard random hold-out for continuity with the earlier baseline.
    random_train, random_test = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE
    )
    random_metrics = evaluate_split(X, y, random_train, random_test, groups)

    # Grouped hold-out prevents the same job-position label appearing in train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    grouped_train, grouped_test = next(splitter.split(X, y, groups=groups))
    grouped_metrics = evaluate_split(X, y, grouped_train, grouped_test, groups)
    grouped_metrics["test_job_positions"] = int(groups.iloc[grouped_test].nunique())

    # Refit on every cleaned training row only after the hold-out evaluation is complete.
    final_model = build_pipeline()
    final_model.fit(X, y)

    metrics = {
        "model": "TF-IDF + engineered skill tokens + Ridge Regression",
        "dataset_rows": int(len(df)),
        "target": "matched_score",
        "random_state": RANDOM_STATE,
        "random_holdout": random_metrics,
        "job_position_group_holdout": grouped_metrics,
        "final_model_features": int(len(final_model.named_steps["tfidf"].vocabulary_)),
        "feature_notes": [
            "TF-IDF word and bigram features",
            "explicit skill coverage bucket tokens",
            "matched skill count token",
            "target excluded from all features",
        ],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {MODEL_PATH}")
    return metrics


def main() -> None:
    """Parse CLI arguments and run training."""
    parser = argparse.ArgumentParser(description="Train the resume ranking model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    train(args.data)


if __name__ == "__main__":
    main()
