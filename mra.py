import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

print("Working directory:", os.getcwd())
print(os.listdir())  # Show files

# 1. Load the dataset (CSV with Excel encoding)
df = pd.read_csv("IMDB Dataset.csv", encoding="latin1")
print(df.head())
print(df.columns)

# Convert sentiment text → numbers
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# 3. Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# 5. Evaluate
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# 6. Try a custom review
review = ["The movie was amazing and emotional"]
rev_vec = vectorizer.transform(review)
result = model.predict(rev_vec)[0]

print("Prediction:", "Positive" if result == 1 else "Negative")
