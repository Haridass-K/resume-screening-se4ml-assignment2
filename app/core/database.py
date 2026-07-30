import sqlite3
from datetime import datetime

import pandas as pd

from app.core.config import DATABASE_DIR, DATABASE_PATH, MAX_HISTORY_RECORDS
from app.core.logger import get_logger

logger = get_logger(__name__)

DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Create and return a SQLite database connection."""

    try:
        return sqlite3.connect(DATABASE_PATH)

    except sqlite3.Error:
        logger.exception(
            "Failed to connect to database: %s",
            DATABASE_PATH,
        )
        raise


def initialize_database() -> None:
    """Create the prediction history table if it does not exist."""

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    resume_text TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    confidence REAL NOT NULL
                )
                """)

            conn.commit()

        logger.info("Database initialized successfully.")

    except sqlite3.Error:
        logger.exception("Database initialization failed.")
        raise


def save_prediction(
    resume_text: str,
    predicted_category: str,
    confidence: float,
) -> None:
    """Save a resume prediction to the database."""

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO prediction_history (
                    timestamp,
                    resume_text,
                    predicted_category,
                    confidence
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    resume_text,
                    predicted_category,
                    float(confidence),
                ),
            )

            conn.commit()

        logger.info(
            "Prediction saved successfully. Category: %s",
            predicted_category,
        )

    except sqlite3.Error:
        logger.exception("Failed to save prediction.")
        raise


def get_prediction_history(
    limit: int = MAX_HISTORY_RECORDS,
) -> pd.DataFrame:
    """Retrieve the latest prediction history."""

    if not isinstance(limit, int) or limit < 1:
        logger.warning(
            "Invalid prediction history limit: %s",
            limit,
        )
        raise ValueError("Limit must be a positive integer.")

    query = """
        SELECT
            id,
            timestamp,
            resume_text,
            predicted_category,
            confidence
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
    """

    try:
        with get_connection() as conn:
            history = pd.read_sql_query(
                query,
                conn,
                params=(limit,),
            )

        logger.info(
            "Retrieved %s prediction history records.",
            len(history),
        )

        return history

    except Exception:
        logger.exception("Failed to retrieve prediction history.")
        raise
