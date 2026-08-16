"""Clean and validate the Kaggle resume-ranking dataset.

Usage:
    python -m ml.data_cleaning --input data/raw/resume_data_for_ranking.csv

The script deliberately keeps the raw dataset unchanged and writes a cleaned
copy plus a JSON quality report under data/processed and data/reports.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


TARGET_COLUMN = "matched_score"


def normalize_column_name(column: str) -> str:
    """Convert a source column name into a stable Python-friendly name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", column.strip()).strip("_").lower()


def normalize_text(value: object) -> str:
    """Normalize whitespace while preserving meaningful resume text."""
    if pd.isna(value):
        return ""

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_dataset(input_path: Path, output_path: Path, report_path: Path) -> dict:
    """Clean the raw dataset and write a reproducible quality report."""
    df = pd.read_csv(input_path)

    original_rows, original_columns = df.shape
    original_duplicates = int(df.duplicated().sum())

    # Normalize column names once so downstream ML code can use stable names.
    df.columns = [normalize_column_name(column) for column in df.columns]

    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for column in text_columns:
        df[column] = df[column].map(normalize_text)

    # Remove only exact duplicate records; do not aggressively discard rows.
    df = df.drop_duplicates().reset_index(drop=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' was not found")

    # The model predicts a normalized match score, so invalid targets are excluded.
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    invalid_target_mask = (
        df[TARGET_COLUMN].isna()
        | ~df[TARGET_COLUMN].between(0.0, 1.0, inclusive="both")
    )
    invalid_target_count = int(invalid_target_mask.sum())
    df = df.loc[~invalid_target_mask].reset_index(drop=True)

    # Empty text is preferable to invented information for optional resume fields.
    for column in text_columns:
        df[column] = df[column].fillna("")

    final_rows, final_columns = df.shape
    remaining_missing = {
        column: int(value)
        for column, value in df.isna().sum().items()
        if int(value) > 0
    }

    report = {
        "dataset": input_path.name,
        "cleaning_method": {
            "column_name_normalization": True,
            "text_whitespace_normalization": True,
            "exact_duplicate_removal": True,
            "target_validation": "matched_score must be numeric and between 0 and 1",
            "text_missing_values": "filled with empty string",
            "numeric_missing_values": "preserved; no artificial zero imputation",
            "aggressive_row_deletion": False,
        },
        "before": {
            "rows": original_rows,
            "columns": original_columns,
            "exact_duplicate_rows": original_duplicates,
        },
        "cleaning": {
            "duplicates_removed": original_duplicates,
            "invalid_target_rows_removed": invalid_target_count,
        },
        "after": {
            "rows": final_rows,
            "columns": final_columns,
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "remaining_missing_values": remaining_missing,
        },
        "target": {
            "column": TARGET_COLUMN,
            "min": float(df[TARGET_COLUMN].min()),
            "max": float(df[TARGET_COLUMN].max()),
            "mean": float(df[TARGET_COLUMN].mean()),
            "median": float(df[TARGET_COLUMN].median()),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for reproducible local execution."""
    parser = argparse.ArgumentParser(description="Clean the resume-ranking dataset")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw Kaggle CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cleaned_resume_data.csv"),
        help="Path for the cleaned CSV",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/reports/data_cleaning_report.json"),
        help="Path for the JSON cleaning report",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset cleaning and print a concise summary."""
    args = parse_args()
    report = clean_dataset(args.input, args.output, args.report)

    print(f"Original rows : {report['before']['rows']:,}")
    print(f"Cleaned rows  : {report['after']['rows']:,}")
    print(f"Duplicates removed: {report['cleaning']['duplicates_removed']}")
    print(f"Invalid targets removed: {report['cleaning']['invalid_target_rows_removed']}")
    print(f"Remaining missing values: {sum(report['after']['remaining_missing_values'].values())}")
    print(f"Cleaned data : {args.output}")
    print(f"Quality report: {args.report}")


if __name__ == "__main__":
    main()
