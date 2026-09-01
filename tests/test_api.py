"""Smoke tests for the FastAPI application."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """The service health endpoint should be available."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_short_job_description_is_rejected() -> None:
    """The API should reject unusably short job descriptions."""
    response = client.post(
        "/api/analyze",
        data={"job_description": "Python developer"},
        files={"resumes": ("candidate.txt", b"short", "text/plain")},
    )

    assert response.status_code == 400
    assert "30 characters" in response.json()["detail"]
