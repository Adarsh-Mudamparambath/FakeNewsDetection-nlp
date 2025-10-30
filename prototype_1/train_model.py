import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Ensure model directory
os.makedirs("model", exist_ok=True)

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add labels
fake_df["label"] = 1
true_df["label"] = 0

# Combine
df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

# Use title + text
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)
X = df["content"]
y = df["label"]

# TF-IDF with bigrams
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1,2), max_features=50000)
X_tfidf = tfidf.fit_transform(X)

# Save vectorizer
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Models
models = {
    "lr_model.pkl": LogisticRegression(max_iter=2000),
    "dt_model.pkl": DecisionTreeClassifier(),
    "gbc_model.pkl": GradientBoostingClassifier(),
    "rfc_model.pkl": RandomForestClassifier(),
    "svm_model.pkl": LinearSVC(),
    "xgb_model.pkl": xgb.XGBClassifier(eval_metric="logloss")
}

# Train & evaluate
for filename, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{filename.upper()} | Acc: {acc:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | F1: {f1:.2f}")
    with open(f"model/{filename}", "wb") as f:
        pickle.dump(model, f)

# Stacking ensemble
stack = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=2000)),
        ("rf", RandomForestClassifier()),
        ("gbc", GradientBoostingClassifier()),
        ("svm", LinearSVC()),
        ("xgb", xgb.XGBClassifier(eval_metric="logloss"))
    ],
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
with open("model/stack_model.pkl", "wb") as f:
    pickle.dump(stack, f)

print("✅ All models + Stacking saved successfully!")
