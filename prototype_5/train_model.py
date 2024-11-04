# Import necessary libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Text cleaning function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)  # Removes text inside parentheses
    text = re.sub(r'\(Reuters\)', '', text)  # Removes (Reuters) string from the text
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.strip()
    return text

# Save the TF-IDF vectorizer for later use
def save_models():
    if not os.path.exists('model'):
        os.makedirs('model')

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    df_fake = pd.read_csv("data/Fake.csv")
    df_true = pd.read_csv("data/True.csv")
    
    # Replace Reuters references and clean the text
    df_true["text"] = df_true["text"].replace("(Reuters)", "", regex=True)
    df_fake["text"] = df_fake["text"].replace("(Reuters)", "", regex=True)

    df_fake['target'] = 0  # Fake news label
    df_true['target'] = 1  # True news label

    # Concatenate datasets
    df = pd.concat([df_fake, df_true], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    # Clean the text
    df["text"] = df["text"].apply(wordopt)

    # Drop rows that are completely empty after text processing
    df = df[df['text'].str.strip() != '']
    df = df.dropna(subset=['text'])

    X = df['text']
    y = df['target']
    X_tfidf = tfidf.fit_transform(X)

    # Save vectorizer
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Initialize models
    lr_model = LogisticRegression()
    dt_model = DecisionTreeClassifier()
    gbc_model = GradientBoostingClassifier()
    rfc_model = RandomForestClassifier()

    # Train models
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    gbc_model.fit(X_train, y_train)
    rfc_model.fit(X_train, y_train)

    # Save models
    with open('model/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('model/dt_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    with open('model/gbc_model.pkl', 'wb') as f:
        pickle.dump(gbc_model, f)
    with open('model/rfc_model.pkl', 'wb') as f:
        pickle.dump(rfc_model, f)

# Load the TF-IDF vectorizer and models if they exist, else train and save them
if not os.path.exists('model/vectorizer.pkl'):
    save_models()

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
    # Clean the input text
    news = wordopt(news)
    
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

# Vectorization to print shapes of training and testing data
if __name__ == "__main__":
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    df_fake = pd.read_csv("data/Fake.csv")
    df_true = pd.read_csv("data/True.csv")
    
    df_fake["text"] = df_fake["text"].replace("(Reuters)", "", regex=True)
    df_true["text"] = df_true["text"].replace("(Reuters)", "", regex=True)
    
    df_fake['target'] = 0  # Fake news label
    df_true['target'] = 1  # True news label

    # Concatenate datasets
    df = pd.concat([df_fake, df_true], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    # Clean the text
    df["text"] = df["text"].apply(wordopt)

    X = df['text']
    y = df['target']
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(X_train)
    xv_test = vectorization.transform(x_test)
    print(xv_train.shape)
    print(xv_test.shape)
