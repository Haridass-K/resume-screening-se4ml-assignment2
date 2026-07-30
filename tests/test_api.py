from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Check that the root endpoint is accessible."""

    response = client.get("/")

    assert response.status_code == 200


def test_health_endpoint():
    """Check that the health endpoint is accessible."""

    response = client.get("/health")

    assert response.status_code == 200
