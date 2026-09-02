"""Smoke and validation tests for the FastAPI application."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """The service health endpoint should be available."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "ml_model_available" in response.json()


def test_short_job_description_is_rejected() -> None:
    """The API should reject unusably short job descriptions."""
    response = client.post(
        "/api/analyze",
        data={"job_description": "Python developer"},
        files={"resumes": ("candidate.txt", b"short", "text/plain")},
    )

    assert response.status_code == 400
    assert "30 characters" in response.json()["detail"]


def test_unsupported_resume_type_is_rejected() -> None:
    """Only PDF, DOCX, and TXT resumes should be accepted."""
    job = "We need a Python developer with SQL and machine learning experience."
    response = client.post(
        "/api/analyze",
        data={"job_description": job},
        files={"resumes": ("candidate.csv", b"python,sql", "text/csv")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_valid_text_resume_is_ranked() -> None:
    """A readable TXT resume should produce an explainable ranking result."""
    job = (
        "We need a Python developer with SQL and machine learning experience "
        "for backend data applications and API development."
    )
    resume = (
        "Python developer with machine learning and SQL experience. "
        "Built FastAPI services and data analysis pipelines. "
        "Experienced in developing production APIs and working with teams."
    )

    response = client.post(
        "/api/analyze",
        data={"job_description": job},
        files={"resumes": ("candidate.txt", resume.encode(), "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_count"] == 1
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["score"] >= 0
    assert "matched_skills" in payload["results"][0]
    assert "missing_skills" in payload["results"][0]
