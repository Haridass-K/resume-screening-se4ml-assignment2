from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from app.ml.predict import predict_resume


def test_model_overfits_small_training_batch():
    """Verify that the model can learn a very small training dataset."""

    texts = [
        "Python machine learning data analysis",
        "Python developer SQL artificial intelligence",
        "Recruitment employee relations human resources",
        "HR manager payroll recruitment",
    ]

    labels = [
        "INFORMATION-TECHNOLOGY",
        "INFORMATION-TECHNOLOGY",
        "HR",
        "HR",
    ]

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(texts, labels)

    training_accuracy = model.score(texts, labels)

    assert training_accuracy == 1.0


def test_inference_output_shape_and_range():
    """Check prediction structure, output size, and confidence range."""

    resume_text = (
        "Experienced Python developer with machine learning, SQL, "
        "data analysis, and software development skills."
    )

    result = predict_resume(resume_text, top_k=3)

    assert isinstance(result["predicted_category"], str)
    assert 0.0 <= result["confidence"] <= 100
    assert len(result["top_predictions"]) == 3

    for prediction in result["top_predictions"]:
        assert isinstance(prediction["category"], str)
        assert 0.0 <= prediction["confidence"] <= 100


def test_prediction_case_and_spacing_invariance():
    """Check that case and extra spaces do not change the prediction."""

    normal_text = "Experienced Python developer with machine learning and SQL skills."

    modified_text = (
        "EXPERIENCED   PYTHON   DEVELOPER WITH MACHINE LEARNING AND SQL SKILLS."
    )

    normal_result = predict_resume(normal_text, top_k=3)
    modified_result = predict_resume(modified_text, top_k=3)

    assert normal_result["predicted_category"] == modified_result["predicted_category"]
