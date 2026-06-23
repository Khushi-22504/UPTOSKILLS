# app.py

import streamlit as st
import joblib
import matplotlib.pyplot as plt

from preprocess import preprocess_text

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("💬 AI Sentiment Analyzer")

st.write(
    "Enter any review and get sentiment prediction."
)

user_text = st.text_area(
    "Enter Text"
)

if st.button("Analyze Sentiment"):

    clean_text = preprocess_text(user_text)

    vector = vectorizer.transform([clean_text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    if prediction == "positive":
        st.success("😊 Positive")

    elif prediction == "negative":
        st.error("😡 Negative")

    else:
        st.info("😐 Neutral")

    st.subheader("Confidence Score")

    fig, ax = plt.subplots()

    ax.bar(
        model.classes_,
        probabilities
    )

    ax.set_ylim([0,1])

    st.pyplot(fig)
