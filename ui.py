import streamlit as st
import joblib
import os

# File paths
MODEL_FILE = "cmnts.pkl"
VECTORIZER_FILE = "vector.pkl"

def load_model_and_vectorizer():
    if not os.path.exists(MODEL_FILE):
        st.error(f"Model file '{MODEL_FILE}' not found. Please make sure it's in the app directory.")
        return None, None
    if not os.path.exists(VECTORIZER_FILE):
        st.error(f"Vectorizer file '{VECTORIZER_FILE}' not found. Please make sure it's in the app directory.")
        return None, None

    ml = joblib.load(MODEL_FILE)
    vec = joblib.load(VECTORIZER_FILE)
    return ml, vec

# Load model and vectorizer
ml, vec = load_model_and_vectorizer()

# Sentiment mapping
sentiment = {
    1: "Positive feedback",
    0: "Negative feedback"
}

st.title("Comments Analyzer")

# Optional rating slider (not used for prediction)
st.slider("Rate us", 1, 10, 7)

# Comment input box
cmnt = st.text_area("Enter your comments")

if st.button("Analyze sentiments"):
    if ml is None or vec is None:
        st.warning("Cannot analyze because the model or vectorizer files are missing.")
    elif not cmnt.strip():
        st.warning("Please enter a comment before analyzing.")
    else:
        v = vec.transform([cmnt])
        prd = ml.predict(v)[0]
        fb = sentiment.get(prd, "Unknown sentiment")
        st.write(f"Sentiment: **{fb}**")
