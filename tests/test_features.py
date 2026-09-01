"""Unit tests for resume/job feature engineering."""

from ml.features import build_pair_text_from_strings, extract_skills, skill_overlap_features


def test_extract_skills_handles_multiword_terms() -> None:
    """Known multiword skills should be extracted as complete phrases."""
    text = "Python, machine learning, SQL and data analysis"
    skills = extract_skills(text)

    assert "python" in skills
    assert "machine learning" in skills
    assert "data analysis" in skills


def test_skill_overlap_features_are_bounded() -> None:
    """Skill coverage must remain in the 0-1 interval."""
    features = skill_overlap_features(
        "Python machine learning SQL",
        "Python machine learning SQL Docker",
    )

    assert features["matched_skill_count"] == 3
    assert features["job_skill_count"] == 4
    assert features["skill_coverage"] == 0.75


def test_training_and_inference_representation_contains_same_markers() -> None:
    """Inference should use the same engineered token format as training."""
    pair = build_pair_text_from_strings(
        "Python developer with machine learning experience",
        "Need Python and machine learning skills",
    )

    assert "[job]" in pair
    assert "skill_coverage_high" in pair
    assert "matched_skills_2" in pair
