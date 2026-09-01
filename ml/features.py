"""Feature engineering shared by training and inference.

The model uses a transparent hybrid representation:
1. TF-IDF captures resume/job language.
2. Skill-overlap tokens expose explicit matching signal to the linear model.

No target information is used here.
"""

from __future__ import annotations

import ast
import re
from typing import Iterable

import pandas as pd

# Vocabulary is intentionally compact and explainable for an academic MVP.
SKILL_VOCABULARY = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust",
    "sql", "postgresql", "mysql", "mongodb", "redis", "html", "css", "react",
    "next.js", "node.js", "express", "fastapi", "flask", "django", "spring",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "keras", "opencv", "spacy", "transformers", "git", "github", "docker",
    "kubernetes", "aws", "azure", "gcp", "linux", "rest api", "graphql",
    "microservices", "power bi", "tableau", "excel", "data analysis", "data science",
    "statistics", "communication", "leadership", "problem solving", "teamwork",
    "agile", "scrum", "project management", "sales", "troubleshooting", "customer service",
    "marketing", "accounting", "maintenance", "budgeting", "testing", "financial statements",
    "quality assurance", "automation", "documentation", "installation",
}


def normalize_text(text: object) -> str:
    """Normalize whitespace and punctuation for consistent matching."""
    if pd.isna(text):
        return ""
    value = str(text).lower().replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", value).strip()


def parse_listish(value: object) -> list[str]:
    """Parse nested/list-like dataset fields without executing arbitrary code."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        result: list[str] = []
        for item in parsed:
            result.extend(parse_listish(item))
        return result

    return [part.strip() for part in re.split(r"[,;|\n]+", text) if part.strip()]


def extract_skills(text: str) -> set[str]:
    """Extract known skills using phrase-aware boundary matching."""
    normalized = normalize_text(text)
    found: set[str] = set()
    for skill in sorted(SKILL_VOCABULARY, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9+#.-]){re.escape(skill)}(?![a-z0-9+#.-])"
        if re.search(pattern, normalized):
            found.add(skill)
    return found


def join_fields(row: pd.Series, columns: Iterable[str]) -> str:
    """Combine available fields while safely ignoring missing values."""
    values = []
    for column in columns:
        value = row.get(column, "")
        if pd.notna(value):
            values.append(str(value))
    return " ".join(values)


def build_resume_text(row: pd.Series) -> str:
    """Build candidate-side text from normalized dataset columns."""
    columns = [
        "career_objective", "skills", "degree_names", "major_field_of_studies",
        "professional_company_names", "positions", "responsibilities",
        "languages", "certification_skills",
    ]
    return join_fields(row, columns)


def build_job_text(row: pd.Series) -> str:
    """Build job-side text from normalized dataset columns."""
    columns = [
        "job_position_name", "educational_requirements", "experiencere_requirement",
        "responsibilities_1", "skills_required", "related_skils_in_job",
    ]
    return join_fields(row, columns)


def skill_overlap_features(resume_text: str, job_text: str) -> dict[str, float]:
    """Calculate explicit skill coverage features for one resume/job pair."""
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    matched = resume_skills & job_skills
    required = len(job_skills)
    coverage = len(matched) / required if required else 0.0
    return {
        "resume_skill_count": float(len(resume_skills)),
        "job_skill_count": float(required),
        "matched_skill_count": float(len(matched)),
        "skill_coverage": coverage,
    }


def structured_tokens(resume_text: str, job_text: str) -> str:
    """Encode continuous overlap features as stable tokens for TF-IDF."""
    features = skill_overlap_features(resume_text, job_text)
    coverage = features["skill_coverage"]
    if coverage >= 0.80:
        bucket = "HIGH"
    elif coverage >= 0.50:
        bucket = "MEDIUM"
    elif coverage >= 0.25:
        bucket = "LOW"
    else:
        bucket = "VERY_LOW"

    return (
        f" SKILL_COVERAGE_{bucket}"
        f" MATCHED_SKILLS_{int(features['matched_skill_count'])}"
    )


def build_pair_text(row: pd.Series) -> str:
    """Build the training representation used by the supervised model."""
    resume = build_resume_text(row)
    job = build_job_text(row)
    return f"{normalize_text(resume)} [JOB] {normalize_text(job)}{structured_tokens(resume, job)}"


def build_pair_text_from_strings(resume_text: str, job_text: str) -> str:
    """Build the exact inference representation used during model training."""
    return (
        f"{normalize_text(resume_text)} [JOB] {normalize_text(job_text)}"
        f"{structured_tokens(resume_text, job_text)}"
    )
