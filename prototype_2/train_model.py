# Import necessary libraries
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Add labels to the datasets
fake_df['label'] = 1  # Fake news label
true_df['label'] = 0  # True news label

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data

# Preprocess the data
X = df['text']
y = df['label']

# Convert text data into TF-IDF features
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# Save the TF-IDF vectorizer for later use
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize the models
lr_model = LogisticRegression()
dt_model = DecisionTreeClassifier()
gbc_model = GradientBoostingClassifier()
rfc_model = RandomForestClassifier()

# Train the models
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
gbc_model.fit(X_train, y_train)
rfc_model.fit(X_train, y_train)

# Evaluate and display accuracy for each model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Logistic Regression accuracy
lr_acc = evaluate_model(lr_model, X_test, y_test)
print(f'Logistic Regression Accuracy: {lr_acc * 100:.2f}%')

# Decision Tree accuracy
dt_acc = evaluate_model(dt_model, X_test, y_test)
print(f'Decision Tree Accuracy: {dt_acc * 100:.2f}%')

# Gradient Boosting Classifier accuracy
gbc_acc = evaluate_model(gbc_model, X_test, y_test)
print(f'Gradient Boosting Classifier Accuracy: {gbc_acc * 100:.2f}%')

# Random Forest Classifier accuracy
rfc_acc = evaluate_model(rfc_model, X_test, y_test)
print(f'Random Forest Classifier Accuracy: {rfc_acc * 100:.2f}%')

# Save the trained models
with open('model/lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open('model/dt_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
with open('model/gbc_model.pkl', 'wb') as f:
    pickle.dump(gbc_model, f)
with open('model/rfc_model.pkl', 'wb') as f:
    pickle.dump(rfc_model, f)
