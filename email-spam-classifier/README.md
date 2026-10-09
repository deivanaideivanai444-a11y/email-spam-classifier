# Email Spam Classifier

A Flask application for comparing TF-IDF spam classifiers, testing messages, and reviewing actual held-out evaluation results. The included UCI SMS Spam Collection contains text messages, not full emails; this project labels its results as an **SMS-based prototype**. Replace it with a labeled email dataset to evaluate real email performance.

## Requirements

- Windows 10/11
- Python 3.10 or later (Python 3.12 recommended)
- VS Code with the Python extension
- Internet access for the first dependency install and UCI dataset download

## Setup in PowerShell

Open the project folder in VS Code, open **Terminal > New Terminal**, and run:

```powershell
cd "C:\Users\deiva\email spam\email-spam-classifier"
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is unavailable, install Python 3.12 from python.org, or use `py -m venv .venv` with another supported Python version.

## Dataset and training

The UCI dataset is downloaded into `data\uci-sms-spam\SMSSpamCollection`. If it is not already there, download the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), extract the archive, and place the file named `SMSSpamCollection` in that folder. Then run:

```powershell
python train_model.py
```

Training uses a fixed, stratified 80/20 train/test split. Each TF-IDF vectorizer is fit only on the training portion; model selection uses the held-out spam-class F1-score, with accuracy as a tie-breaker. The trained classifier, vectorizer, actual comparison metrics, and confusion-matrix image are saved under `models\`. Training fails with an actionable error if the dataset or required columns are missing; no sample metrics are fabricated.

For a CSV dataset, set the exact column names before training:

```powershell
$env:SPAM_TEXT_COLUMN = "email_body"
$env:SPAM_LABEL_COLUMN = "category"
python train_model.py --dataset "data\labeled_emails.csv"
```

Recognized labels include `spam`/`ham`, `junk`/`legitimate`, `not spam`, and `1`/`0`. Rows with missing text or labels are discarded. Both classes must have at least two records.

## Run and test

```powershell
python -m pytest -q
python app.py
```

Open **http://127.0.0.1:5000**. The model is loaded once when Flask starts. After retraining, restart Flask to load the new artifacts. Prediction history keeps only a short preview with email addresses and URLs redacted, plus the prediction, confidence, timestamp, and message length. The database is `database\predictions.sqlite3`.

## API

- `GET /api/health` reports app and model readiness.
- `POST /api/predict` accepts `{"text": "..."}` and returns a label, spam probability, confidence, and model name.
- `GET /api/stats` reports prediction counts and recent daily totals.
- `GET /api/history` returns recent redacted previews; `DELETE /api/history` clears them.
- `GET /api/metrics` returns comparison and confusion-matrix values computed on the held-out test set.

Until a dataset has been trained, health reports `model_unavailable`; prediction and metrics return HTTP 503. The SMS model is a learning prototype and is not a substitute for an email-specific model or production security filtering.