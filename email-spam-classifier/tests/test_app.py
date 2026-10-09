import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from app import create_app
from train_model import load_dataset


@pytest.fixture
def app_instance(tmp_path):
    database_path = tmp_path / "predictions.sqlite3"
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    metrics_path = tmp_path / "metrics.json"
    vectorizer = TfidfVectorizer()
    samples = ["free prize claim now", "win cash offer", "meeting moved to monday", "please review the report"]
    labels = ["spam", "spam", "ham", "ham"]
    features = vectorizer.fit_transform(samples)
    model = MultinomialNB().fit(features, labels)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    metrics_path.write_text(
        json.dumps(
            {
                "best_model": "Test Naive Bayes",
                "models": {"Test Naive Bayes": {"accuracy": 1, "precision": 1, "recall": 1, "f1_score": 1}},
                "best": {"accuracy": 1, "precision": 1, "recall": 1, "f1_score": 1, "confusion_matrix": [[2, 0], [0, 2]]},
                "important_features": {"spam": [], "ham": []},
                "dataset_type": "test fixture",
            }
        ),
        encoding="utf-8",
    )
    return create_app(
        {
            "TESTING": True,
            "DATABASE_PATH": str(database_path),
            "MODEL_PATH": str(model_path),
            "VECTORIZER_PATH": str(vectorizer_path),
            "METRICS_PATH": str(metrics_path),
        }
    )


@pytest.fixture
def client(app_instance):
    return app_instance.test_client()


def test_health_reports_loaded_model(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["model_available"] is True
    assert response.json["model_name"] == "Test Naive Bayes"


def test_predict_stores_redacted_preview_and_stats(client):
    response = client.post(
        "/api/predict",
        json={"text": "Congratulations, claim your free prize at https://example.test now"},
    )
    assert response.status_code == 200
    assert response.json["prediction"] == "spam"
    assert 0 <= response.json["spam_probability"] <= 1
    history = client.get("/api/history").json["items"]
    assert len(history) == 1
    assert "https://example.test" not in history[0]["preview"]
    assert client.get("/api/stats").json["total_predictions"] == 1


@pytest.mark.parametrize("payload", [None, {}, {"text": 4}, {"text": "   "}])
def test_predict_rejects_invalid_input(client, payload):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "error" in response.json


def test_metrics_return_saved_test_results(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert response.json["best"]["confusion_matrix"] == [[2, 0], [0, 2]]


def test_history_can_be_cleared(client):
    client.post("/api/predict", json={"text": "please review the report"})
    response = client.delete("/api/history")
    assert response.status_code == 200
    assert client.get("/api/stats").json["total_predictions"] == 0


def test_history_rejects_invalid_limit(client):
    assert client.get("/api/history?limit=500").status_code == 400


@pytest.mark.parametrize("path", ["/", "/classify", "/dashboard", "/history", "/performance"])
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"Signal Desk" in response.data


def test_untrained_model_never_returns_a_prediction(tmp_path):
    client = create_app(
        {
            "TESTING": True,
            "DATABASE_PATH": str(tmp_path / "predictions.sqlite3"),
            "MODEL_PATH": str(tmp_path / "missing-model.joblib"),
            "VECTORIZER_PATH": str(tmp_path / "missing-vectorizer.joblib"),
            "METRICS_PATH": str(tmp_path / "missing-metrics.json"),
        }
    ).test_client()
    assert client.get("/api/health").json["model_available"] is False
    assert client.post("/api/predict", json={"text": "free prize"}).status_code == 503
    assert client.get("/api/metrics").status_code == 503


def test_csv_text_and_label_columns_are_configurable(tmp_path):
    dataset = tmp_path / "messages.csv"
    dataset.write_text(
        "body,category\nfree prize,spam\nmeeting moved,ham\nclaim reward,spam\nsee you monday,ham\n",
        encoding="utf-8",
    )
    frame = load_dataset(dataset, text_column="body", label_column="category")
    assert frame["label"].tolist() == ["spam", "ham", "spam", "ham"]
    assert frame["text"].tolist()[0] == "free prize"