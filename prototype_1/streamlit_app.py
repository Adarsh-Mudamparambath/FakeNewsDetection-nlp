import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
lr_model = joblib.load('models/Logistic_Regression.pkl')
dt_model = joblib.load('models/Decision_Tree.pkl')
gb_model = joblib.load('models/Gradient_Boosting.pkl')
rf_model = joblib.load('models/Random_Forest.pkl')

# Load vectorizer
vectorizer = joblib.load('models/vectorizer.pkl')

# Function to predict news category
def predict_news(model, text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return 'Fake News' if prediction[0] == 1 else 'Real News'

# Streamlit App Interface
st.title("Fake News Detection")

news_text = st.text_area("Enter the news text to classify:")

if st.button("Classify News"):
    if news_text.strip() != "":
        # Predict using different models
        lr_prediction = predict_news(lr_model, news_text)
        dt_prediction = predict_news(dt_model, news_text)
        gb_prediction = predict_news(gb_model, news_text)
        rf_prediction = predict_news(rf_model, news_text)

        # Display results
        st.write(f"**Logistic Regression Prediction:** {lr_prediction}")
        st.write(f"**Decision Tree Prediction:** {dt_prediction}")
        st.write(f"**Gradient Boosting Prediction:** {gb_prediction}")
        st.write(f"**Random Forest Prediction:** {rf_prediction}")
    else:
        st.warning("Please enter some news text for classification.")
