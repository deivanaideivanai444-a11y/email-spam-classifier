# 📧 Email Spam Classifier Using Machine Learning

## 📌 Project Overview

The Email Spam Classifier is a Machine Learning project that identifies whether a message is **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and Machine Learning algorithms.

The system analyzes the text entered by the user and predicts whether the message is spam or legitimate.

## 🎯 Objectives

* Automatically classify spam and legitimate messages.
* Apply NLP techniques to process text data.
* Train and evaluate Machine Learning models.
* Provide a simple and user-friendly interface for message classification.

## ✨ Features

* Spam and Ham message classification.
* Text preprocessing using TF-IDF.
* Machine Learning-based predictions.
* Model performance evaluation.
* Interactive user interface.
* Fast prediction of user-provided text.

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn
* **Natural Language Processing:** TF-IDF
* **Data Processing:** Pandas, NumPy
* **Backend:** Flask
* **Frontend:** HTML, CSS, JavaScript
* **Database:** SQLite (if implemented)
* **Visualization:** Matplotlib, Seaborn (if implemented)

## 🤖 Machine Learning Algorithms

* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)

The algorithms can be evaluated to select a suitable model for classification.

## 📂 Project Structure

```text
email-spam-classifier/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── templates/
└── static/
    ├── css/
    └── js/
```

*Note: The structure above is an example. Update it to match the actual files in the repository.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/deivanaideivanai444-a11y/email-spam-classifier.git
```

### 2. Open the Project Folder

```bash
cd email-spam-classifier
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Prepare the Dataset

Download a suitable labeled spam dataset and place it in the location expected by the training script. Check the dataset filename and column names before training.

### 7. Train the Machine Learning Model

```bash
python train_model.py
```

Run this command if your project includes `train_model.py` and the required dataset is available.

### 8. Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## 📊 Expected Outcome

The application is designed to classify input messages as Spam or Ham. Actual performance depends on the dataset, model, and evaluation results.

## 🔮 Future Enhancements

* Support for real email datasets.
* Phishing and scam message detection.
* Model performance visualization.
* Email integration.
* Improved text preprocessing and model evaluation.

## 👩‍💻 Author

**Deivanai**

B.Tech – Artificial Intelligence and Data Science

## 📜 License

This project is intended for educational and learning purposes.
