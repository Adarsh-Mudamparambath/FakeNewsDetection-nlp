# Import necessary libraries
import streamlit as st
import pickle

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

# Prediction function for each model
def predict_news(news, model):
    vectorized_input = vectorizer.transform([news])
    prediction = model.predict(vectorized_input)
    return prediction

# Streamlit interface
st.title("Fake News Detection App")

# Input text area for user to enter the news
news = st.text_area("Enter the news text to classify:")

# If the classify button is clicked
if st.button("Classify"):
    # Get predictions from all models
    lr_pred = predict_news(news, lr_model)
    dt_pred = predict_news(news, dt_model)
    gbc_pred = predict_news(news, gbc_model)
    rfc_pred = predict_news(news, rfc_model)

    # Convert the results into human-readable labels
    def get_label(pred):
        return "Fake" if pred == 1 else "True"

    # Display predictions from all models
    st.write(f"**Logistic Regression Prediction**: {get_label(lr_pred)}")
    st.write(f"**Decision Tree Prediction**: {get_label(dt_pred)}")
    st.write(f"**Gradient Boosting Classifier Prediction**: {get_label(gbc_pred)}")
    st.write(f"**Random Forest Classifier Prediction**: {get_label(rfc_pred)}")
