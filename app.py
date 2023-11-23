import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load your pickled model
with open('model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)  # Replace 'your_model.pkl' with the actual filename

# Load your TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)  # Replace 'your_vectorizer.pkl' with the actual filename

# Output variable representing possible sections
cl = ['world', 'opinion', 'business', 'podcasts', 'us', 'lifestyle', 'arts']

# Streamlit app
st.title("Article Section Prediction")

# Get user input
user_input = st.text_input("Enter the news article title:")

if user_input:
    # Vectorize the input text
    input_vectorized = vectorizer.transform([user_input])

    # Make a prediction
    prediction = classifier.predict(input_vectorized)[0]

    # Map numerical prediction back to section label
    predicted_section = cl[prediction]

    # Display the prediction result
    st.subheader(f"Predicted Section: {predicted_section}")
