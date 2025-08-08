import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# --- Step 1: Example Training Data ---
data = {
    "comment": [
        "very good",
        "excellent service",
        "I am happy with this",
        "bad experience",
        "worst service ever",
        "I hate this product",
        "amazing and wonderful",
        "not satisfied",
        "great work",
        "poor quality"
    ],
    "label": [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative
}

df = pd.DataFrame(data)

# --- Step 2: Vectorize text ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["comment"])
y = df["label"]

# --- Step 3: Train model ---
model = MultinomialNB()
model.fit(X, y)

# --- Step 4: Save model & vectorizer ---
joblib.dump(model, "cmnts.pkl")
joblib.dump(vectorizer, "vector.pkl")

print("✅ Model and vectorizer saved successfully!")
