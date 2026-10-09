import argparse
import html
import json
import os
import re
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import DATA_DIR, MODEL_DIR


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")
LABEL_ALIASES = {
    "spam": "spam",
    "junk": "spam",
    "1": "spam",
    "ham": "ham",
    "legitimate": "ham",
    "not spam": "ham",
    "non-spam": "ham",
    "0": "ham",
}


def clean_text(value: object) -> str:
    text = html.unescape(str(value))
    text = TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" urltoken ", text)
    return " ".join(text.split())


def load_dataset(path: Path, text_column: str = "text", label_column: str = "label") -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download the UCI SMS Spam Collection and place "
            "SMSSpamCollection in data/uci-sms-spam/."
        )

    if path.name.lower() == "smsspamcollection" or path.suffix.lower() in {".tsv", ".tab"}:
        frame = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
    else:
        frame = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
        missing_columns = [column for column in (text_column, label_column) if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Dataset is missing configured column(s): {', '.join(missing_columns)}. "
                f"Available columns: {', '.join(map(str, frame.columns))}. "
                "Set SPAM_TEXT_COLUMN and SPAM_LABEL_COLUMN to match your CSV."
            )
        frame = frame[[label_column, text_column]].rename(columns={label_column: "label", text_column: "text"})

    frame = frame.dropna(subset=["label", "text"]).copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    frame = frame.dropna(subset=["label"])
    frame["text"] = frame["text"].map(clean_text)
    frame = frame[frame["text"].str.len() > 0]
    if frame.empty:
        raise ValueError("The dataset has no usable labeled text rows after removing missing or invalid values.")
    if set(frame["label"].unique()) != {"ham", "spam"}:
        raise ValueError("The dataset must contain both recognized ham and spam labels.")
    if frame["label"].value_counts().min() < 2:
        raise ValueError("At least two examples of each class are required for a stratified train/test split.")
    return frame.reset_index(drop=True)


def _new_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def _top_features(pipeline: Pipeline, limit: int = 12) -> dict[str, list[dict[str, float | str]]]:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    weight_model = classifier
    if hasattr(classifier, "calibrated_classifiers_"):
        weight_model = classifier.calibrated_classifiers_[0].estimator
    terms = vectorizer.get_feature_names_out()
    classes = list(weight_model.classes_)
    spam_index = classes.index("spam")
    ham_index = classes.index("ham")

    if hasattr(weight_model, "feature_log_prob_"):
        weights = weight_model.feature_log_prob_[spam_index] - weight_model.feature_log_prob_[ham_index]
    elif hasattr(weight_model, "coef_"):
        weights = weight_model.coef_
        if spam_index == 0:
            weights = -weights
    else:
        return {"spam": [], "ham": []}

    if hasattr(weights, "toarray"):
        weights = weights.toarray()
    weights = np.asarray(weights).ravel()
    spam_indexes = np.argsort(weights)[-limit:][::-1]
    ham_indexes = np.argsort(weights)[:limit]
    return {
        "spam": [{"word": str(terms[index]), "weight": round(float(weights[index]), 4)} for index in spam_indexes],
        "ham": [{"word": str(terms[index]), "weight": round(float(-weights[index]), 4)} for index in ham_indexes],
    }


def _save_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="PuBu",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        cbar=False,
        linewidths=1,
        linecolor="white",
        ax=axis,
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title("Best model · held-out test set")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train(dataset_path: Path, text_column: str, label_column: str, model_dir: Path) -> dict:
    frame = load_dataset(dataset_path, text_column=text_column, label_column=label_column)
    training_text, testing_text, training_labels, testing_labels = train_test_split(
        frame["text"],
        frame["label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["label"],
    )
    fitting_text, validation_text, fitting_labels, validation_labels = train_test_split(
        training_text,
        training_labels,
        test_size=0.25,
        random_state=43,
        stratify=training_labels,
    )

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=SVC(kernel="linear", class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=3,
            ensemble=False,
        ),
    }
    validation_scores: dict[str, dict[str, float]] = {}
    for name, classifier in candidates.items():
        validation_pipeline = _new_pipeline(clone(classifier))
        validation_pipeline.fit(fitting_text, fitting_labels)
        validation_predictions = validation_pipeline.predict(validation_text)
        validation_scores[name] = {
            "accuracy": round(float(accuracy_score(validation_labels, validation_predictions)), 4),
            "f1_score": round(float(f1_score(validation_labels, validation_predictions, pos_label="spam", zero_division=0)), 4),
        }

    best_name = max(
        validation_scores,
        key=lambda name: (validation_scores[name]["f1_score"], validation_scores[name]["accuracy"]),
    )
    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}

    for name, classifier in candidates.items():
        pipeline = _new_pipeline(clone(classifier))
        pipeline.fit(training_text, training_labels)
        predictions = pipeline.predict(testing_text)
        matrix = confusion_matrix(testing_labels, predictions, labels=["ham", "spam"])
        results[name] = {
            "accuracy": round(float(accuracy_score(testing_labels, predictions)), 4),
            "precision": round(float(precision_score(testing_labels, predictions, pos_label="spam", zero_division=0)), 4),
            "recall": round(float(recall_score(testing_labels, predictions, pos_label="spam", zero_division=0)), 4),
            "f1_score": round(float(f1_score(testing_labels, predictions, pos_label="spam", zero_division=0)), 4),
            "confusion_matrix": matrix.tolist(),
        }
        fitted[name] = pipeline

    best_pipeline = fitted[best_name]
    metrics = {
        "best_model": best_name,
        "selection_metric": "Validation spam-class F1-score, with validation accuracy as the tie-breaker",
        "dataset_name": "UCI SMS Spam Collection" if dataset_path.name.lower() == "smsspamcollection" else dataset_path.name,
        "dataset_type": "SMS-based prototype; metrics do not establish full-email performance",
        "dataset_rows_used": int(len(frame)),
        "training_rows": int(len(training_text)),
        "selection_training_rows": int(len(fitting_text)),
        "validation_rows": int(len(validation_text)),
        "test_rows": int(len(testing_text)),
        "test_size": 0.2,
        "random_state": 42,
        "selection_random_state": 43,
        "label_order": ["ham", "spam"],
        "selection_scores": validation_scores,
        "models": results,
        "best": results[best_name],
        "important_features": _top_features(best_pipeline),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    model_target = model_dir / "best_model.joblib"
    vectorizer_target = model_dir / "tfidf_vectorizer.joblib"
    metrics_target = model_dir / "metrics.json"
    matrix_target = model_dir / "confusion_matrix.png"
    with tempfile.TemporaryDirectory(dir=model_dir) as temporary_directory:
        temporary = Path(temporary_directory)
        joblib.dump(best_pipeline.named_steps["classifier"], temporary / model_target.name)
        joblib.dump(best_pipeline.named_steps["tfidf"], temporary / vectorizer_target.name)
        (temporary / metrics_target.name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _save_confusion_matrix(results[best_name]["confusion_matrix"], temporary / matrix_target.name)
        os.replace(temporary / model_target.name, model_target)
        os.replace(temporary / vectorizer_target.name, vectorizer_target)
        os.replace(temporary / metrics_target.name, metrics_target)
        os.replace(temporary / matrix_target.name, matrix_target)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare spam classifiers on a labeled dataset.")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "uci-sms-spam" / "SMSSpamCollection")
    parser.add_argument("--text-column", default=os.environ.get("SPAM_TEXT_COLUMN", "text"))
    parser.add_argument("--label-column", default=os.environ.get("SPAM_LABEL_COLUMN", "label"))
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    metrics = train(args.dataset, args.text_column, args.label_column, args.model_dir)
    print(f"Dataset: {metrics['dataset_name']} ({metrics['dataset_rows_used']:,} usable messages)")
    print(f"Holdout: {metrics['training_rows']:,} train / {metrics['test_rows']:,} test")
    for name, scores in metrics["models"].items():
        print(
            f"{name}: accuracy={scores['accuracy']:.3f}  precision={scores['precision']:.3f}  "
            f"recall={scores['recall']:.3f}  F1={scores['f1_score']:.3f}"
        )
    print(f"Selected: {metrics['best_model']}")
    print(f"Saved model, vectorizer, genuine holdout metrics, and confusion matrix to {args.model_dir}")


if __name__ == "__main__":
    main()