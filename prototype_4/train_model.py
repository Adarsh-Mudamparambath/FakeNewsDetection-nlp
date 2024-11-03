import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

# Create the 'model' folder if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load datasets
true_news = pd.read_csv('data/True.csv')
fake_news = pd.read_csv('data/Fake.csv')

# Label the data
true_news['label'] = 0
fake_news['label'] = 1

# Combine and shuffle
data = pd.concat([true_news, fake_news], axis=0).sample(frac=1).reset_index(drop=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train the Gradient Boosting model
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = gb_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Gradient Boosting Model Accuracy: {accuracy * 100:.2f}%')

# Save the model and vectorizer
with open('model/gb_classifier.pkl', 'wb') as f:
    pickle.dump(gb_classifier, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
