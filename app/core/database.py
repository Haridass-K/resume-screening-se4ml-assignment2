import sqlite3
from datetime import datetime

import pandas as pd

from app.core.config import DATABASE_DIR, DATABASE_PATH
from app.core.logger import get_logger

logger = get_logger(__name__)

DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    """Create a SQLite database connection."""
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    """Create the prediction history table if it does not exist."""

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                resume_text TEXT,
                predicted_category TEXT,
                confidence REAL
            )
        """)

        conn.commit()

    logger.info("Database initialized successfully.")


def save_prediction(resume_text, predicted_category, confidence):
    """Save a prediction to the database."""

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO prediction_history
            (timestamp, resume_text, predicted_category, confidence)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            resume_text,
            predicted_category,
            confidence,
        ))

        conn.commit()

    logger.info("Prediction saved successfully.")


def get_prediction_history(limit=100):
    """Retrieve the latest prediction history."""

    with get_connection() as conn:

        query = f"""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT {limit}
        """

        history = pd.read_sql(query, conn)

    return history