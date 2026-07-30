from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_prediction_api_integration():
    """Check the complete prediction workflow."""

    response = client.post(
        "/predict",
        json={
            "resume_text": (
                "Experienced Python developer with machine learning, "
                "SQL, and cloud computing skills."
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_category" in result
    assert "confidence" in result
    assert "top_predictions" in result
