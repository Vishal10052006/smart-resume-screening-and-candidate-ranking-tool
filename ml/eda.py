"""Run exploratory data analysis for the resume-ranking dataset.

Usage:
    python -m ml.eda --data data/processed/cleaned_resume_data.csv

The script creates reproducible text/CSV reports under data/reports.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "cleaned_resume_data.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "reports"
TARGET = "matched_score"

RESUME_TEXT_COLUMNS = [
    "career_objective", "skills", "degree_names", "major_field_of_studies",
    "professional_company_names", "positions", "responsibilities",
    "languages", "certification_skills",
]

JOB_TEXT_COLUMNS = [
    "job_position_name", "educational_requirements", "experiencere_requirement",
    "responsibilities_1", "skills_required", "related_skils_in_job",
]


def parse_listish(value: object) -> list[str]:
    """Parse list-like fields while retaining a safe text fallback."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [item.strip() for item in re.split(r"[,;\n|]+", text) if item.strip()]


def build_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Combine available text columns into one analysis representation."""
    available = [column for column in columns if column in df.columns]
    return df[available].fillna("").astype(str).agg(" ".join, axis=1)


def run_eda(data_path: Path, report_dir: Path) -> dict:
    """Calculate EDA statistics and write reproducible reports."""
    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found")
    if df[TARGET].isna().any():
        raise ValueError("Cleaned data contains missing target values")

    resume_text = build_text(df, RESUME_TEXT_COLUMNS)
    job_text = build_text(df, JOB_TEXT_COLUMNS)
    resume_skills = df["skills"].map(parse_listish)
    job_skills = df["skills_required"].map(parse_listish)

    skill_frequency: dict[str, int] = {}
    for skills in resume_skills:
        for skill in skills:
            key = re.sub(r"\s+", " ", skill.lower().strip())
            if key:
                skill_frequency[key] = skill_frequency.get(key, 0) + 1

    score_bins = pd.cut(
        df[TARGET],
        bins=[-0.001, 0.20, 0.40, 0.60, 0.80, 1.00],
        labels=["0.00-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"],
    )

    analysis = pd.DataFrame({
        "resume_words": resume_text.str.findall(r"\b\w+\b").str.len(),
        "job_words": job_text.str.findall(r"\b\w+\b").str.len(),
        "resume_skill_count": resume_skills.map(len),
        "job_skill_count": job_skills.map(len),
        TARGET: df[TARGET].astype(float),
    })

    correlations = (
        analysis.corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )

    report = {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "job_positions": int(df["job_position_name"].nunique()),
        },
        "target": {
            "name": TARGET,
            "min": float(df[TARGET].min()),
            "max": float(df[TARGET].max()),
            "mean": float(df[TARGET].mean()),
            "median": float(df[TARGET].median()),
            "std": float(df[TARGET].std()),
            "unique_values": int(df[TARGET].nunique()),
            "score_bands": {
                str(label): int(count)
                for label, count in score_bins.value_counts(sort=False).items()
            },
        },
        "text": {
            "resume_words": {
                "mean": float(analysis["resume_words"].mean()),
                "median": float(analysis["resume_words"].median()),
                "min": int(analysis["resume_words"].min()),
                "max": int(analysis["resume_words"].max()),
            },
            "job_words": {
                "mean": float(analysis["job_words"].mean()),
                "median": float(analysis["job_words"].median()),
                "min": int(analysis["job_words"].min()),
                "max": int(analysis["job_words"].max()),
            },
        },
        "skills": {
            "resume_count_mean": float(analysis["resume_skill_count"].mean()),
            "resume_count_median": float(analysis["resume_skill_count"].median()),
            "job_count_mean": float(analysis["job_skill_count"].mean()),
            "job_count_median": float(analysis["job_skill_count"].median()),
            "top_20": dict(sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)[:20]),
        },
        "job_positions": {
            "top_10": {
                str(name): int(count)
                for name, count in df["job_position_name"].value_counts().head(10).items()
            }
        },
        "numeric_correlations_with_target": {
            name: float(value) for name, value in correlations.items()
        },
        "modeling_notes": [
            "matched_score is a continuous regression target.",
            "Resume and job text are suitable for TF-IDF/NLP features.",
            "Skill counts and resume/job skill overlap are candidates for structured features.",
            "matched_score must never be included in the feature matrix.",
            "Job positions repeat heavily, so role-aware validation is recommended.",
            "The near-uniform job-position representation suggests deliberate balancing or construction; external validation is recommended before production claims.",
        ],
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    job_counts = df["job_position_name"].value_counts()
    pd.DataFrame({"job_position": job_counts.index, "records": job_counts.values}).to_csv(
        report_dir / "job_position_distribution.csv", index=False
    )

    band_counts = score_bins.value_counts(sort=False)
    pd.DataFrame({"score_band": band_counts.index.astype(str), "records": band_counts.values}).to_csv(
        report_dir / "match_score_distribution.csv", index=False
    )

    top_skills = sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)[:50]
    pd.DataFrame(top_skills, columns=["skill", "records"]).to_csv(
        report_dir / "top_skills.csv", index=False
    )
    return report


def main() -> None:
    """Run EDA from the command line."""
    parser = argparse.ArgumentParser(description="Run resume dataset EDA")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    report = run_eda(args.data, args.report_dir)
    print(f"Rows: {report['dataset']['rows']:,}")
    print(f"Columns: {report['dataset']['columns']}")
    print(f"Job positions: {report['dataset']['job_positions']}")
    print(f"Target mean: {report['target']['mean']:.4f}")
    print(f"Target median: {report['target']['median']:.4f}")
    print(f"Reports written to: {args.report_dir}")


if __name__ == "__main__":
    main()
