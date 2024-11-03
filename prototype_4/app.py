import streamlit as st
import pickle

# Load the model and vectorizer
with open('model/gb_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit app title
st.title('Fake News Detection App')

# Input for the news text
user_input = st.text_area('Enter news text here:')

# Button to predict
if st.button('Predict'):
    if user_input.strip():
        # Vectorize the user input
        user_input_tfidf = vectorizer.transform([user_input])

        # Predict using the model
        prediction = model.predict(user_input_tfidf)[0]

        # Display the result
        if prediction == 0:
            st.write("This news is **Real**.")
        else:
            st.write("This news is **Fake**.")
    else:
        st.write("Please enter some text for prediction.")
