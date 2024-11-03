import streamlit as st
import joblib
import pandas as pd

# Load the model and vectorizer
model = joblib.load('voting_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Title of the app
st.title('Fake News Detection')

# User input for news article
user_input = st.text_area("Enter the news article:")

if st.button("Predict"):
    if user_input:
        # Vectorize the user input
        user_input_tfidf = tfidf.transform([user_input])
        
        # Predictions from individual models
        logistic_prob = model.estimators_[0].predict_proba(user_input_tfidf)[0][1]
        random_forest_prob = model.estimators_[1].predict_proba(user_input_tfidf)[0][1]
        svc_prob = model.estimators_[2].predict_proba(user_input_tfidf)[0][1]

        # Combined prediction using voting classifier
        voting_result = model.predict([user_input_tfidf])
        combined_prob = model.predict_proba(user_input_tfidf)[0][1]

        # Display the results
        st.write(f"**Logistic Regression Probability of being True:** {logistic_prob:.2f}")
        st.write(f"**Random Forest Probability of being True:** {random_forest_prob:.2f}")
        st.write(f"**Support Vector Classifier Probability of being True:** {svc_prob:.2f}")
        st.write(f"**Combined Prediction Probability of being True:** {combined_prob:.2f}")
        
        if voting_result[0] == 1:
            st.write("The article is **TRUE**.")
        else:
            st.write("The article is **FAKE**.")
    else:
        st.warning("Please enter some text for prediction.")
