# CineScore — Movie Review Sentiment Analyzer

> AI-powered sentiment analysis for movie reviews. Paste any review and get an instant Positive/Negative prediction with confidence score.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange) ![Accuracy](https://img.shields.io/badge/Accuracy-89%25-brightgreen)

---

## Demo

[![CineScore UI](https://i.imgur.com/placeholder.png)](https://cinescore-jek9.onrender.com)

- Type or paste any movie review
- Click **Analyze Sentiment**
- Get a Positive / Negative result with confidence %

---

## Features

- **3-model comparison** — Logistic Regression, Naive Bayes, LinearSVC (best is auto-selected)
- **TF-IDF vectorization** — unigrams + bigrams, 20,000 features
- **89%+ accuracy** on IMDB 50K dataset
- **REST API** — `/api/predict` endpoint for programmatic use
- **Cinema-themed UI** — dark mode, animated confidence bar, example chips

---

## Project Structure

```
movie-review-analysis/
├── train.py              ← ML training pipeline
├── app.py                ← Flask web server
├── templates/
│   └── index.html        ← Frontend UI
├── static/
│   ├── style.css
│   └── script.js
├── models/               ← Auto-created by train.py (git-ignored)
│   ├── vectorizer.pkl
│   └── classifier.pkl
├── IMDB Dataset.csv      ← Dataset (see below)
├── requirements.txt
└── .gitignore
```

---

## Dataset

This project uses the **IMDB 50K Movie Reviews** dataset.

**Download it here:**
[https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

1. Download `IMDB Dataset.csv` from Kaggle (free account required)
2. Place it in the project root (same folder as `train.py`)

> The dataset is not included in this repo because it exceeds GitHub's file size recommendations.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/movie-review-analysis.git
cd movie-review-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Follow the [Dataset](#dataset) instructions above and place `IMDB Dataset.csv` in the project root.

### 4. Train the model

```bash
python train.py
```

This will:
- Load and vectorize the dataset
- Train and compare 3 models
- Print accuracy, F1 score, and confusion matrix
- Save the best model to `models/`

Expected output:
```
Model                      Accuracy   F1 Score
------------------------  ---------- ----------
Logistic Regression          0.8926     0.8923
Naive Bayes                  0.8737     0.8724
Linear SVM                   0.8929     0.8922  <-- best
```

### 5. Start the web server

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## API

### `POST /api/predict`

Predict sentiment for a review.

**Request:**
```json
{
  "review": "This film was an absolute masterpiece."
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 87.9,
  "label": 1
}
```

### `GET /api/health`

```json
{
  "status": "ok",
  "model": "LinearSVC"
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Web framework | Flask 3.1 |
| ML library | scikit-learn 1.5 |
| Vectorization | TF-IDF (unigrams + bigrams) |
| Models | Logistic Regression, Naive Bayes, LinearSVC |
| Frontend | HTML / CSS / Vanilla JS |
| Dataset | IMDB 50K Movie Reviews (Kaggle) |

---

## License

MIT
