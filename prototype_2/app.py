import streamlit as st
import pickle
import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ==============================
# Load Classical Models + Vectorizer
# ==============================
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

model_files = {
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "Gradient Boosting": "gbc_model.pkl",
    "Random Forest": "rfc_model.pkl",
    "SVM": "svm_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "Stacked Ensemble": "stack_model.pkl",
    "DistilBERT": "distilbert_model"   # special case
}

models = {}
for name, file in model_files.items():
    if name == "DistilBERT":
        tokenizer = DistilBertTokenizerFast.from_pretrained(f"model/{file}")
        models[name] = DistilBertForSequenceClassification.from_pretrained(f"model/{file}")
    else:
        with open(f"model/{file}", "rb") as f:
            models[name] = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "DistilBERT" in models:
    models["DistilBERT"].to(device)

# ==============================
# Prediction helpers
# ==============================
def predict(news, model_choice):
    if model_choice == "DistilBERT":
        inputs = tokenizer(news, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = models["DistilBERT"](**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        return pred, probs[1]  # probability of Fake
    else:
        vectorized = vectorizer.transform([news])
        model = models[model_choice]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vectorized)[0][1]
        else:  # LinearSVC
            decision = model.decision_function(vectorized)
            prob = 1 / (1 + np.exp(-decision))[0]
        pred = model.predict(vectorized)[0]
        return pred, prob

def lime_predict(texts):
    vec = vectorizer.transform(texts)
    return models[model_choice].predict_proba(vec)

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detection - NLP")
st.markdown("### Detect whether a news article is **Fake** or **True** using ML & NLP")

st.sidebar.title("⚙️ Options")
model_choice = st.sidebar.selectbox("Select Model", list(model_files.keys()))
st.sidebar.info("Choose from Classical ML models or DistilBERT.")

# ---- Single Text Classification ----
st.subheader("✍️ Test Single News Article")
news = st.text_area("Enter news content here:", height=150)

if st.button("Classify"):
    if not news.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        pred, prob = predict(news, model_choice)
        label = "Fake" if pred == 1 else "True"
        color = "red" if pred == 1 else "green"

        st.markdown(f"### ✅ Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.progress(int(prob * 100))
        st.write(f"**Probability of being Fake:** {prob*100:.2f}%")

        # Explainability only for classical ML models
        if model_choice != "DistilBERT" and hasattr(models[model_choice], "predict_proba"):
            explainer = LimeTextExplainer(class_names=["True", "Fake"])
            exp = explainer.explain_instance(
                news,
                lime_predict,
                num_features=10,
                labels=[1]
            )
            st.markdown("### 🔍 Words influencing prediction:")
            for word, weight in exp.as_list(label=1):
                word_color = "red" if weight > 0 else "green"
                st.markdown(f"- <span style='color:{word_color}'>{word}</span> ({weight:.3f})", unsafe_allow_html=True)

# ---- Bulk Classification ----
st.subheader("📂 Bulk News Classification")
uploaded_file = st.file_uploader("Upload CSV (must have 'title' and 'text')", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)
        preds, probs = [], []
        for content in df["content"]:
            pred, prob = predict(content, model_choice)
            preds.append("Fake" if pred == 1 else "True")
            probs.append(f"{prob*100:.2f}%")
        df["Prediction"] = preds
        df["Fake_Probability"] = probs
        st.dataframe(df[["title", "Prediction", "Fake_Probability"]])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "classified_news.csv", "text/csv")
    else:
        st.error("CSV must have 'title' and 'text' columns")

st.markdown("---")
st.caption("Powered by ML + DistilBERT 🚀")
