from app.core.config import TOP_K_PREDICTIONS
from app.core.logger import get_logger
from app.ml.model_loader import load_model
from app.ml.preprocess import clean_resume

logger = get_logger(__name__)


def predict_resume(
    resume_text: str,
    top_k: int = TOP_K_PREDICTIONS,
) -> dict:
    """Predict the most likely resume categories."""

    cleaned_text = clean_resume(resume_text)

    if not cleaned_text:
        logger.warning("Received an empty resume after preprocessing.")
        raise ValueError("Resume text cannot be empty.")

    model = load_model()

    if not hasattr(model, "predict_proba"):
        logger.error("Production model does not support probability prediction.")
        raise AttributeError(
            "The production model does not support probability prediction."
        )

    classes = model.classes_

    if not 1 <= top_k <= len(classes):
        logger.warning(
            "Invalid top_k value: %s. Expected between 1 and %s.",
            top_k,
            len(classes),
        )
        raise ValueError(f"top_k must be between 1 and {len(classes)}.")

    try:
        predicted_category = model.predict([cleaned_text])[0]
        probabilities = model.predict_proba([cleaned_text])[0]

        ranked_results = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        top_predictions = [
            {
                "category": str(category),
                "confidence": round(float(probability) * 100, 2),
            }
            for category, probability in ranked_results
        ]

        result = {
            "predicted_category": str(predicted_category),
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
        }

        logger.info(
            "Prediction completed successfully. Category: %s",
            predicted_category,
        )

        return result

    except Exception:
        logger.exception("Resume prediction failed.")
        raise
