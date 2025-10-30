import streamlit as st
import pickle
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

# ==============================
# Load vectorizer
# ==============================
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ==============================
# Load models
# ==============================
model_files = {
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "Gradient Boosting": "gbc_model.pkl",
    "Random Forest": "rfc_model.pkl",
    "SVM": "svm_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "Stacked Ensemble": "stack_model.pkl"
}
models = {}
for name, file in model_files.items():
    with open(f"model/{file}", "rb") as f:
        models[name] = pickle.load(f)

# ==============================
# Prediction helper
# ==============================
def predict(news, model):
    vectorized = vectorizer.transform([news])
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vectorized)[0][1]
    else:  # SVM LinearSVC has no predict_proba
        decision = model.decision_function(vectorized)
        prob = 1 / (1 + np.exp(-decision))[0]
    pred = model.predict(vectorized)[0]
    return pred, prob

# ==============================
# Wrapper for LIME
# ==============================
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
st.sidebar.info("Choose any ML model or the Stacked Ensemble.")

# ---- Single Text Classification ----
st.subheader("✍️ Test Single News Article")
news = st.text_area("Enter news content here:", height=150)

if st.button("Classify"):
    if not news.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        pred, prob = predict(news, models[model_choice])
        label = "Fake" if pred == 1 else "True"
        color = "red" if pred == 1 else "green"

        st.markdown(f"### ✅ Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.progress(int(prob * 100))
        st.write(f"**Probability of being Fake:** {prob*100:.2f}%")

        # Explainability with LIME
        if hasattr(models[model_choice], "predict_proba"):
            explainer = LimeTextExplainer(class_names=["True", "Fake"])
            exp = explainer.explain_instance(
                news,
                lime_predict,     # ✅ fixed wrapper
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
            pred, prob = predict(content, models[model_choice])
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
st.caption("Powered by NLP 🚀")
