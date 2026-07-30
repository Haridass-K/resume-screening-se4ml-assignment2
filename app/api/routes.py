from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.database import save_prediction
from app.core.logger import get_logger
from app.ml.predict import predict_resume
from app.security.input_validator import validate_resume_input

logger = get_logger(__name__)

router = APIRouter()


class ResumeRequest(BaseModel):
    resume_text: str = Field(
        ...,
        min_length=20,
        description="Resume text used for category prediction.",
    )


class PredictionItem(BaseModel):
    category: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_predictions: list[PredictionItem]


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=200,
    responses={
        400: {"description": "Invalid request."},
        500: {"description": "Internal server error."},
    },
)
def predict(request: ResumeRequest) -> PredictionResponse:
    """Predict the resume category and save the result."""

    try:
        logger.info("Received prediction request.")
        validate_resume_input(request.resume_text)

        result = predict_resume(request.resume_text)

        save_prediction(
            resume_text=request.resume_text,
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
        )

        return PredictionResponse(**result)

    except ValueError as error:
        logger.warning(
            "Invalid prediction request: %s",
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception("Prediction API failed due to an unexpected error.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        ) from error
