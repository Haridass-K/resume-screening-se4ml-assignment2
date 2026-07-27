from pathlib import Path

# Project root folder
BASE_DIR = Path(__file__).resolve().parents[2]

# Main folders
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
LOG_DIR = BASE_DIR / "logs"
QUALITY_RESULTS_DIR = BASE_DIR / "quality_results"

# Main files
DATA_PATH = DATA_DIR / "Resume.csv"
MODEL_PATH = MODEL_DIR / "resume_screening_model.pkl"
DATABASE_PATH = DATABASE_DIR / "prediction_history.db"
LOG_PATH = LOG_DIR / "application.log"

# Dataset columns
TEXT_COLUMN = "Resume_str"
TARGET_COLUMN = "Category"

# Application settings
APP_NAME = "AI-Powered Resume Screening System"
MODEL_VERSION = "2.0"
TOP_K_PREDICTIONS = 3
MAX_HISTORY_RECORDS = 100