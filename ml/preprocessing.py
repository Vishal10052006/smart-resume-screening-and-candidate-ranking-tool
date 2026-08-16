"""Shared text preparation for training and inference."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


# These columns represent candidate-side information available in the dataset.
RESUME_COLUMNS = [
    "career_objective",
    "skills",
    "degree_names",
    "major_field_of_studies",
    "professional_company_names",
    "positions",
    "responsibilities",
    "certification_skills",
]

# These columns represent job-side information available in the dataset.
JOB_COLUMNS = [
    "job_position_name",
    "educationaL_requirements",
    "experiencere_requirement",
    "responsibilities.1",
    "skills_required",
    "related_skils_in_job",
]


def join_fields(row: pd.Series, columns: Iterable[str]) -> str:
    """Combine available fields while safely ignoring missing values."""
    values = []
    for column in columns:
        value = row.get(column, "")
        if pd.notna(value):
            values.append(str(value))
    return " ".join(values)


def build_resume_text(row: pd.Series) -> str:
    """Build the candidate-side text representation."""
    return join_fields(row, RESUME_COLUMNS)


def build_job_text(row: pd.Series) -> str:
    """Build the job-side text representation."""
    return join_fields(row, JOB_COLUMNS)


def build_pair_text(row: pd.Series) -> str:
    """Build one supervised-learning example from resume and job text."""
    return f"{build_resume_text(row)} [JOB] {build_job_text(row)}".lower()
