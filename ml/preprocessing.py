"""Shared preprocessing entry points for the ML pipeline.

The actual feature logic lives in `ml.features` so training and API inference
cannot silently drift apart.
"""

from __future__ import annotations

from ml.features import (
    build_job_text,
    build_pair_text,
    build_pair_text_from_strings,
    build_resume_text,
    extract_skills,
    parse_listish,
    skill_overlap_features,
)

__all__ = [
    "build_job_text",
    "build_pair_text",
    "build_pair_text_from_strings",
    "build_resume_text",
    "extract_skills",
    "parse_listish",
    "skill_overlap_features",
]
