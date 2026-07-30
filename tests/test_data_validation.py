import pytest

from app.ml.predict import predict_resume


def test_invalid_top_k_zero():
    """Check that top_k cannot be zero."""

    with pytest.raises(ValueError):
        predict_resume(
            "Experienced Python developer with machine learning skills.",
            top_k=0,
        )


def test_invalid_top_k_too_large():
    """Check that top_k cannot exceed the number of classes."""

    with pytest.raises(ValueError):
        predict_resume(
            "Experienced Python developer with machine learning skills.",
            top_k=100,
        )
