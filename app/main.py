from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import APP_NAME, MODEL_VERSION
from app.core.database import initialize_database
from app.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources."""

    initialize_database()
    logger.info("Application started successfully.")

    yield

    logger.info("Application stopped.")


app = FastAPI(
    title=APP_NAME,
    version=MODEL_VERSION,
    description="Production API for AI-Powered Resume Screening",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
def home() -> dict:
    """Home endpoint."""

    return {"message": f"{APP_NAME} API is running."}


@app.get("/health")
def health() -> dict:
    """Health endpoint."""

    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
    }
