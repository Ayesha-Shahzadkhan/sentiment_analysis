import streamlit as st
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# UI
st.title("Sentiment Analysis")

tweet = st.text_input("Enter a tweet:")

if tweet:
    text_vec = vectorizer.transform([tweet])
    prediction = model.predict(text_vec)[0]
    st.success(f"Predicted Sentiment: {prediction}")
