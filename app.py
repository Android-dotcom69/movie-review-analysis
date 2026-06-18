import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

MODEL_DIR = os.path.join(BASE_DIR, "models")
vectorizer = None
classifier = None


def load_model():
    global vectorizer, classifier
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    clf_path = os.path.join(MODEL_DIR, "classifier.pkl")

    if not os.path.exists(vec_path) or not os.path.exists(clf_path):
        raise FileNotFoundError("Model files not found. Run `python train.py` first.")

    vectorizer = joblib.load(vec_path)
    classifier = joblib.load(clf_path)
    print(f"Model loaded: {type(classifier).__name__}")


def get_confidence(model, vec):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        label = int(np.argmax(proba))
        confidence = float(proba[label])
    else:
        # LinearSVC: use decision_function + sigmoid
        score = model.decision_function(vec)[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))
        label = 1 if score > 0 else 0
    return label, confidence


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = (data or {}).get("review", "").strip()

    if not review:
        return jsonify({"error": "No review text provided."}), 400
    if len(review) < 10:
        return jsonify({"error": "Review is too short. Please write at least 10 characters."}), 400

    vec = vectorizer.transform([review])
    label, confidence = get_confidence(classifier, vec)

    return jsonify({
        "sentiment": "positive" if label == 1 else "negative",
        "confidence": round(confidence * 100, 1),
        "label": label,
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": type(classifier).__name__})


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, host="0.0.0.0")
