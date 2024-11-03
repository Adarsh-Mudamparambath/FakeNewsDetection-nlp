import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # Corrected import
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
fake_data = pd.read_csv('data/fake.csv')
true_data = pd.read_csv('data/true.csv')

# Prepare the data
fake_data['label'] = 0  # Fake news label
true_data['label'] = 1  # True news label

data = pd.concat([fake_data, true_data])
X = data['text']  # Adjust the column name if different
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize the models
logistic = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier()
svc = SVC(probability=True)

# Create a voting classifier
model = VotingClassifier(estimators=[
    ('logistic', logistic),
    ('random_forest', random_forest),
    ('svc', svc)
], voting='soft')

# Train the model
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, 'voting_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')
