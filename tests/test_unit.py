import pytest

from app.ml.predict import predict_resume
from app.ml.preprocessing import clean_resume


def test_clean_resume():
    """Check that resume text is cleaned correctly."""

    text = "Python Developer EMAIL@test.com https://example.com"
    cleaned_text = clean_resume(text)

    assert "python developer" in cleaned_text
    assert "email@test.com" not in cleaned_text
    assert "https://example.com" not in cleaned_text


def test_clean_resume_empty_text():
    """Check that empty text remains empty after preprocessing."""

    assert clean_resume("") == ""


def test_predict_resume_returns_expected_structure():
    """Check that prediction returns the expected output structure."""

    resume_text = (
        "Experienced software engineer with Python, SQL, "
        "machine learning, and cloud development skills."
    )

    result = predict_resume(resume_text)

    assert "predicted_category" in result
    assert "confidence" in result
    assert "top_predictions" in result

    assert isinstance(result["predicted_category"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["top_predictions"], list)


def test_predict_resume_empty_input():
    """Check that an empty resume raises a ValueError."""

    with pytest.raises(ValueError):
        predict_resume("")
