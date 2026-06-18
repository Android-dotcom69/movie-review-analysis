import zipfile
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

DATASET_PATH = "IMDB Dataset.csv"
MODEL_DIR = "models"


def load_dataset(path):
    # The file is a ZIP64 archive (may be truncated). Try standard zip first,
    # then fall back to raw deflate decompression of the local file entry.
    try:
        with zipfile.ZipFile(path, "r") as z:
            with z.open(z.namelist()[0]) as f:
                return pd.read_csv(f, encoding="latin1")
    except (zipfile.BadZipFile, KeyError):
        pass

    # Raw deflate: parse local file header to find data offset
    import struct, zlib, io
    with open(path, "rb") as f:
        header = f.read(30)
        fn_len = struct.unpack_from("<H", header, 26)[0]
        ex_len = struct.unpack_from("<H", header, 28)[0]
        f.seek(30 + fn_len + ex_len)
        compressed = f.read()

    dec_obj = zlib.decompressobj(-15)
    try:
        text_bytes = dec_obj.decompress(compressed)
    except zlib.error:
        # Stream is truncated — use whatever was decompressed
        text_bytes = dec_obj.flush()

    text = text_bytes.decode("utf-8", errors="replace")
    # Drop any incomplete last line
    last_newline = text.rfind("\n")
    if last_newline != -1:
        text = text[:last_newline]

    df = pd.read_csv(io.StringIO(text))
    return df


def train():
    print("=" * 50)
    print("  Movie Review Sentiment — Training Pipeline")
    print("=" * 50)

    print("\n[1/4] Loading dataset...")
    df = load_dataset(DATASET_PATH)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    print(f"      {len(df):,} reviews  |  Positive: {df['label'].sum():,}  Negative: {(df['label']==0).sum():,}")

    print("\n[2/4] Vectorizing text (TF-IDF, unigrams + bigrams)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    print("\n[3/4] Training & comparing models...")
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "Linear SVM":          LinearSVC(max_iter=2000, C=1.0),
    }

    best_model, best_acc, best_name = None, 0, ""
    print(f"\n  {'Model':<24} {'Accuracy':>10} {'F1 Score':>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10}")

    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        marker = " <-- best" if acc > best_acc else ""
        print(f"  {name:<24} {acc:>10.4f} {f1:>10.4f}{marker}")
        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name

    print(f"\n  Winner: {best_name} ({best_acc:.4f})")

    preds = best_model.predict(X_test_vec)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))

    cm = confusion_matrix(y_test, preds)
    print("--- Confusion Matrix ---")
    print(f"  True Neg : {cm[0][0]:>5}   False Pos: {cm[0][1]:>5}")
    print(f"  False Neg: {cm[1][0]:>5}   True Pos : {cm[1][1]:>5}")

    print("\n[4/4] Saving model + vectorizer...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(best_model, os.path.join(MODEL_DIR, "classifier.pkl"))
    print(f"      Saved to {MODEL_DIR}/")
    print("\nDone. Run `python app.py` to start the web server.\n")


if __name__ == "__main__":
    train()
