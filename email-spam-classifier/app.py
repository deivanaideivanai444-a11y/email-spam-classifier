import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

from config import Config


EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _connect(database_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _database(database_path: str):
    connection = _connect(database_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _initialize_database(database_path: str) -> None:
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with _database(database_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL CHECK(label IN ('spam', 'ham')),
                spam_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                preview TEXT NOT NULL,
                character_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )


def _private_preview(text: str) -> str:
    preview = EMAIL_PATTERN.sub("[email]", text)
    preview = URL_PATTERN.sub("[link]", preview)
    preview = " ".join(preview.split())
    return preview[:100] + ("..." if len(preview) > 100 else "")


def _load_artifacts(app: Flask) -> None:
    model_path = Path(app.config["MODEL_PATH"])
    vectorizer_path = Path(app.config["VECTORIZER_PATH"])
    metrics_path = Path(app.config["METRICS_PATH"])
    app.extensions["spam_model"] = None
    app.extensions["tfidf_vectorizer"] = None
    app.extensions["model_metrics"] = None
    app.extensions["model_name"] = None
    app.extensions["model_error"] = "Train a model with train_model.py to enable predictions."

    if not (model_path.is_file() and vectorizer_path.is_file() and metrics_path.is_file()):
        return

    try:
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        app.extensions["spam_model"] = joblib.load(model_path)
        app.extensions["tfidf_vectorizer"] = joblib.load(vectorizer_path)
        app.extensions["model_metrics"] = metrics
        app.extensions["model_name"] = metrics["best_model"]
        app.extensions["model_error"] = None
    except (OSError, ValueError, KeyError, EOFError, ImportError) as error:
        app.extensions["spam_model"] = None
        app.extensions["tfidf_vectorizer"] = None
        app.extensions["model_metrics"] = None
        app.extensions["model_name"] = None
        app.extensions["model_error"] = f"Could not load the saved model: {error}"


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    _initialize_database(app.config["DATABASE_PATH"])
    _load_artifacts(app)

    @app.get("/")
    def home():
        return render_template("index.html", active_page="home")

    @app.get("/classify")
    def classify_page():
        return render_template("classifier.html", active_page="classify")

    @app.get("/dashboard")
    def dashboard_page():
        return render_template("dashboard.html", active_page="dashboard")

    @app.get("/history")
    def history_page():
        return render_template("history.html", active_page="history")

    @app.get("/performance")
    def performance_page():
        return render_template("performance.html", active_page="performance")

    @app.get("/api/health")
    def health():
        available = app.extensions["spam_model"] is not None
        return jsonify(
            status="ready" if available else "model_unavailable",
            application="Email Spam Classifier",
            model_available=available,
            model_name=app.extensions["model_name"],
            detail=app.extensions["model_error"],
            dataset="UCI SMS Spam Collection (SMS-based prototype)" if available else None,
        )

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object with a text field."), 400
        text = payload.get("text")
        if not isinstance(text, str):
            return jsonify(error="The text field must be a string."), 400
        text = text.strip()
        if not text:
            return jsonify(error="Enter some message text before predicting."), 400
        if len(text) > app.config["MAX_TEXT_LENGTH"]:
            return jsonify(error=f"Text must be {app.config['MAX_TEXT_LENGTH']:,} characters or fewer."), 400

        model = app.extensions["spam_model"]
        vectorizer = app.extensions["tfidf_vectorizer"]
        if model is None or vectorizer is None:
            return jsonify(error=app.extensions["model_error"]), 503

        try:
            features = vectorizer.transform([text])
            predicted_label = str(model.predict(features)[0]).lower()
            classes = [str(label).lower() for label in model.classes_]
            spam_probability = float(model.predict_proba(features)[0][classes.index("spam")])
            confidence = spam_probability if predicted_label == "spam" else 1.0 - spam_probability
        except (ValueError, IndexError, AttributeError) as error:
            app.logger.exception("Model prediction failed")
            return jsonify(error=f"Prediction failed: {error}"), 500

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _database(app.config["DATABASE_PATH"]) as connection:
            cursor = connection.execute(
                """INSERT INTO predictions
                   (label, spam_probability, confidence, preview, character_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (predicted_label, spam_probability, confidence, _private_preview(text), len(text), created_at),
            )
            prediction_id = cursor.lastrowid

        return jsonify(
            id=prediction_id,
            prediction=predicted_label,
            spam_probability=spam_probability,
            confidence=confidence,
            model=app.extensions["model_name"],
            note="Confidence is the model's estimated probability for its predicted class; it is not a guarantee.",
        )

    @app.get("/api/stats")
    def stats():
        with _database(app.config["DATABASE_PATH"]) as connection:
            rows = connection.execute(
                "SELECT label, COUNT(*) AS count FROM predictions GROUP BY label"
            ).fetchall()
            daily_rows = connection.execute(
                """SELECT substr(created_at, 1, 10) AS day,
                          SUM(CASE WHEN label = 'spam' THEN 1 ELSE 0 END) AS spam,
                          SUM(CASE WHEN label = 'ham' THEN 1 ELSE 0 END) AS ham
                   FROM predictions GROUP BY day ORDER BY day DESC LIMIT 7"""
            ).fetchall()

        counts = {row["label"]: row["count"] for row in rows}
        spam_count = counts.get("spam", 0)
        ham_count = counts.get("ham", 0)
        total = spam_count + ham_count
        daily = [dict(row) for row in reversed(daily_rows)]
        return jsonify(
            total_predictions=total,
            spam_count=spam_count,
            ham_count=ham_count,
            spam_percentage=round(spam_count / total * 100, 1) if total else 0,
            daily=daily,
        )

    @app.get("/api/history")
    def history():
        try:
            limit = int(request.args.get("limit", "50"))
        except ValueError:
            return jsonify(error="limit must be a whole number."), 400
        if not 1 <= limit <= 100:
            return jsonify(error="limit must be between 1 and 100."), 400
        with _database(app.config["DATABASE_PATH"]) as connection:
            rows = connection.execute(
                """SELECT id, label AS prediction, spam_probability, confidence,
                          preview, character_count, created_at
                   FROM predictions ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return jsonify(items=[dict(row) for row in rows])

    @app.delete("/api/history")
    def clear_history():
        with _database(app.config["DATABASE_PATH"]) as connection:
            connection.execute("DELETE FROM predictions")
        return jsonify(message="Prediction history cleared.")

    @app.get("/api/metrics")
    def metrics():
        model_metrics = app.extensions["model_metrics"]
        if model_metrics is None:
            return jsonify(error=app.extensions["model_error"]), 503
        return jsonify(model_metrics)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)