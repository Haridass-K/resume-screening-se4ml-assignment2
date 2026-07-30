from functools import lru_cache

import joblib

from app.core.config import MODEL_PATH
from app.core.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model() -> object:
    """Load and cache the trained resume screening model."""

    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
        return model

    except Exception:
        logger.exception(
            "Failed to load model from %s",
            MODEL_PATH,
        )
        raise
