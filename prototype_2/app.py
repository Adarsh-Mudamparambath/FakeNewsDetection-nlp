# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Load the TF-IDF vectorizer and models
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('model/dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('model/gbc_model.pkl', 'rb') as f:
    gbc_model = pickle.load(f)

with open('model/rfc_model.pkl', 'rb') as f:
    rfc_model = pickle.load(f)

# Prediction function that returns both label and probability
def predict_news(news, model):
    vectorized_input = vectorizer.transform([news])
    prediction = model.predict(vectorized_input)
    probability = model.predict_proba(vectorized_input)[0][1]  # Probability of 'Fake' class (label 1)
    return prediction[0], probability

# Streamlit interface
st.title("Fake News Detection App")

# Input text area for user to enter the news
news = st.text_area("Enter the news text to classify:")

# If the classify button is clicked
if st.button("Classify"):
    # Get predictions and probabilities from all models
    lr_pred, lr_prob = predict_news(news, lr_model)
    dt_pred, dt_prob = predict_news(news, dt_model)
    gbc_pred, gbc_prob = predict_news(news, gbc_model)
    rfc_pred, rfc_prob = predict_news(news, rfc_model)

    # Convert the results into human-readable labels
    def get_label(pred):
        return "Fake" if pred == 1 else "True"

    # Display individual predictions from each model
    st.write(f"**Logistic Regression Prediction**: {get_label(lr_pred)} (Fake : {lr_prob * 100:.2f}%)")
    st.write(f"**Decision Tree Prediction**: {get_label(dt_pred)} (Fake : {dt_prob * 100:.2f}%)")
    st.write(f"**Gradient Boosting Prediction**: {get_label(gbc_pred)} (Fake : {gbc_prob * 100:.2f}%)")
    st.write(f"**Random Forest Prediction**: {get_label(rfc_pred)} (Fake : {rfc_prob * 100:.2f}%)")

    # Calculate the average probability of being fake
    avg_fake_prob = np.mean([lr_prob, dt_prob, gbc_prob, rfc_prob]) * 100

    # Final combined prediction based on majority voting
    fake_votes = sum([lr_pred, dt_pred, gbc_pred, rfc_pred])
    final_prediction = "Fake" if fake_votes >= 2 else "True"

    # Display the final combined prediction and fake probability percentage
    st.write(f"**Combined Prediction (Voting)**: {final_prediction}")
    st.write(f"**Fake News Probability**: {avg_fake_prob:.2f}%")
