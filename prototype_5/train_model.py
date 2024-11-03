import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk as nlp
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV

# Load the datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Load new fake dataset (assuming it's named 'NewFake.csv')
df_new_fake = pd.read_csv("NewFake.csv")

# Rename and align columns as needed
if 'headline' in df_new_fake.columns:
    df_new_fake.rename(columns={'headline': 'title', 'body': 'text'}, inplace=True)

# Add label column for new fake news
df_new_fake['target'] = 0

# Add missing columns if needed (subject, date, etc.)
if 'subject' not in df_new_fake.columns:
    df_new_fake['subject'] = 'unknown'
if 'date' not in df_new_fake.columns:
    df_new_fake['date'] = 'unknown'

# Drop unnecessary columns and align with the existing dataframe structure
df_new_fake = df_new_fake.drop(["title", "subject", "date"], axis=1, errors='ignore')

# Concatenate new dataset with original fake and true datasets
df_fake_combined = pd.concat([df_fake, df_new_fake], axis=0)
df = pd.concat([df_fake_combined, df_true], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

# Remove empty rows after combining dataframes
df.dropna(subset=['text'], inplace=True)

# Text cleaning
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\(.*?\)', '', text)  # Removes text inside parentheses
    text = re.sub('\\W', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.strip()
    return text

df["text"] = df["text"].apply(wordopt)

# Drop rows that are completely empty after text processing
df = df[df['text'].str.strip() != '']
df = df.dropna(subset=['text'])

# Prepare the dataset for training
X = df["text"]
Y = df["target"]

# Train-test split
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Vectorization using TF-IDF
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(X_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression model
lr = LogisticRegression()
lr.fit(xv_train, Y_train)
print("The Accuracy of the Logistic Regression Model is {}".format(lr.score(xv_test, y_test)))
print(classification_report(y_test, lr.predict(xv_test)))

# Decision Tree Classifier model
dtc = DecisionTreeClassifier()
dtc.fit(xv_train, Y_train)
print("The Accuracy of the Decision Tree Classifier Model is {}".format(dtc.score(xv_test, y_test)))
print(classification_report(y_test, dtc.predict(xv_test)))

# Gradient Boosting Classifier model
gclf = GradientBoostingClassifier()
gclf.fit(xv_train, Y_train)
print("The Accuracy of the Gradient Boosting Classifier Model is {}".format(gclf.score(xv_test, y_test)))
print(classification_report(y_test, gclf.predict(xv_test)))

# Hyperparameter tuning for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search_rfc = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search_rfc.fit(xv_train, Y_train)
rclf_best = grid_search_rfc.best_estimator_
print(f"Best Parameters for RandomForest: {grid_search_rfc.best_params_}")

# Random Forest Classifier model with best parameters
print("The Accuracy of the Random Forest Classifier Model is {}".format(rclf_best.score(xv_test, y_test)))
print(classification_report(y_test, rclf_best.predict(xv_test)))

# Combine all the trained models into an ensemble voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('dtc', dtc),
        ('gclf', gclf),
        ('rclf', rclf_best)
    ],
    voting='soft'
)

# Fit the voting classifier
voting_clf.fit(xv_train, Y_train)
voting_accuracy = accuracy_score(y_test, voting_clf.predict(xv_test))
print(f"The Accuracy of the Ensemble Voting Classifier Model is: {voting_accuracy:.2f}")

# Cross-validation to evaluate model performance more robustly
cross_val_scores = cross_val_score(lr, xv_train, Y_train, cv=5)
print(f"Cross-validation scores for Logistic Regression: {cross_val_scores}")
print(f"Average cross-validation score: {np.mean(cross_val_scores):.2f}")

# Manual testing function
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = lr.predict(new_xv_test)
    pred_DT = dtc.predict(new_xv_test)
    pred_GBC = gclf.predict(new_xv_test)
    pred_RFC = rclf_best.predict(new_xv_test)
    pred_ENSEMBLE = voting_clf.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {} \nEnsemble Prediction: {}".format(
        output_lable(pred_LR[0]),
        output_lable(pred_DT[0]),
        output_lable(pred_GBC[0]),
        output_lable(pred_RFC[0]),
        output_lable(pred_ENSEMBLE[0])
    ))
