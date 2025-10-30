import os
import pickle
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ===============================
# Ensure model directory
# ===============================
os.makedirs("model", exist_ok=True)

# ===============================
# Load dataset
# ===============================
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df["label"] = 1
true_df["label"] = 0

df = pd.concat([fake_df, true_df], axis=0).sample(frac=1).reset_index(drop=True)
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

X = df["content"]
y = df["label"]

# ===============================
# Classical ML models
# ===============================
print("\n🔹 Training Classical ML models...")

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

# Train & evaluate each
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

# Stacking
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

print("✅ Classical models + Stacking saved successfully!")

# ===============================
# DistilBERT Training
# ===============================
print("\n🔹 Training DistilBERT model...")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
dataset = Dataset.from_pandas(df[["content", "label"]])

def tokenize(batch):
    return tokenizer(batch["content"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)

# Train/test split for BERT
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
test_dataset = train_test["test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="./model/bert_results",
    eval_strategy="epoch",   # ⚠️ your Transformers version expects eval_strategy
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    logging_dir="./logs",
    load_best_model_at_end=True,
    save_total_limit=1,
    no_cuda=False  # ✅ force GPU if available
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("model/distilbert_model")
tokenizer.save_pretrained("model/distilbert_model")

print("✅ DistilBERT model saved successfully in model/distilbert_model/")
