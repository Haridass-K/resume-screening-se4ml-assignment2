from app.ml.model_loader import load_model


def test_model_loads_successfully():
    """Check that the production model loads correctly."""

    model = load_model()

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
