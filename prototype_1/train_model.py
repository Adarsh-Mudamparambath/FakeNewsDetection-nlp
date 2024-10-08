import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Step 1: Load the Data
fake_df = pd.read_csv('data/Fake.csv')  # Adjust the path based on your structure
true_df = pd.read_csv('data/True.csv')

# Add a column for labels
fake_df['label'] = 0  # Fake news
true_df['label'] = 1  # Real news

# Combine datasets
data = pd.concat([fake_df, true_df], ignore_index=True)

# Step 2: Preprocess the Data
data.dropna(inplace=True)
X = data['text']  # Adjust if your text column is named differently
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train the Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier()
}

for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, f'models/{model_name.replace(" ", "_")}.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'models/vectorizer.pkl')
