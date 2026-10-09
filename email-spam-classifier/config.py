import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "local-development-key-change-me")
    DATABASE_PATH = os.environ.get("DATABASE_PATH", str(DATABASE_DIR / "predictions.sqlite3"))
    MODEL_PATH = os.environ.get("MODEL_PATH", str(MODEL_DIR / "best_model.joblib"))
    VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", str(MODEL_DIR / "tfidf_vectorizer.joblib"))
    METRICS_PATH = os.environ.get("METRICS_PATH", str(MODEL_DIR / "metrics.json"))
    MAX_CONTENT_LENGTH = 1024 * 1024
    MAX_TEXT_LENGTH = 20000