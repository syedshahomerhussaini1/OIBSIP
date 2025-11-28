"""
Spam Email Detection — Enhanced

This script loads a spam dataset, trains a Logistic Regression classifier
using a TF-IDF representation, and produces enhanced visualizations and
analysis including ROC/PR curves, most predictive words, word clouds
and misclassified examples.

Run:
    python spam_detection.py

Requirements:
    pandas, numpy, seaborn, matplotlib, scikit-learn, joblib, wordcloud
"""

import os
import re
import string
import joblib
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    WordCloud = None
    HAS_WORDCLOUD = False

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# Configure plotting style
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 5)


def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


def load_data(path: str = "spam.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]
    return df


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    df["cleaned"] = df["message"].apply(clean_text)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    X_text = df["cleaned"]
    y = df["label_num"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X = vectorizer.fit_transform(X_text)

    return X, y, vectorizer, df


def plot_class_distribution(df: pd.DataFrame, out_dir="plots"):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="label", data=df, palette="muted")
    plt.title("Class distribution")
    plt.xlabel("")
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(height, (p.get_x() + p.get_width() / 2.0, height), ha="center", va="bottom")
    path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, labels=("Ham", "Spam"), out_dir="plots"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_roc_pr_curves(y_test, y_scores, out_dir="plots"):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="blue", lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    path = os.path.join(out_dir, "precision_recall_curve.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_wordclouds(df: pd.DataFrame, out_dir="plots"):
    if not HAS_WORDCLOUD:
        print("WordCloud package not installed — skipping wordcloud generation. To enable, run: pip install wordcloud")
        return
    # Create word clouds for ham and spam
    ham_text = " ".join(df.loc[df["label"] == "ham", "cleaned"].astype(str))
    spam_text = " ".join(df.loc[df["label"] == "spam", "cleaned"].astype(str))

    if len(spam_text.strip()) > 0:
        wc_spam = WordCloud(width=800, height=400, background_color="white").generate(spam_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_spam, interpolation="bilinear")
        plt.axis("off")
        plt.title("Top words in SPAM messages")
        path = os.path.join(out_dir, "wordcloud_spam.png")
        plt.savefig(path, bbox_inches="tight")
        plt.show()

    if len(ham_text.strip()) > 0:
        wc_ham = WordCloud(width=800, height=400, background_color="white").generate(ham_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_ham, interpolation="bilinear")
        plt.axis("off")
        plt.title("Top words in HAM messages")
        path = os.path.join(out_dir, "wordcloud_ham.png")
        plt.savefig(path, bbox_inches="tight")
        plt.show()


def plot_top_predictive_words(vectorizer, model, top_n=20, out_dir="plots"):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    coef_series = pd.Series(coefs, index=feature_names)
    top_spam = coef_series.sort_values(ascending=False).head(top_n)
    top_ham = coef_series.sort_values(ascending=True).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_spam.values, y=top_spam.index, color="red")
    plt.title(f"Top {top_n} Words Associated with SPAM")
    path = os.path.join(out_dir, "top_spam_words.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_ham.values, y=top_ham.index, color="green")
    plt.title(f"Top {top_n} Words Associated with HAM")
    path = os.path.join(out_dir, "top_ham_words.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def show_misclassified_examples(msg_test, y_test, y_pred, out_dir="plots", n=10):
    df_mis = pd.DataFrame({"message": msg_test.reset_index(drop=True), "actual": y_test.reset_index(drop=True), "pred": y_pred})
    mis = df_mis[df_mis["actual"] != df_mis["pred"]].copy()
    # Map actual/pred back to label names
    mis["actual_label"] = mis["actual"].map({0: "ham", 1: "spam"})
    mis["pred_label"] = mis["pred"].map({0: "ham", 1: "spam"})
    if mis.shape[0] == 0:
        print("No misclassified examples (nice!).")
        return
    # Show and save a few
    print("\n===== SAMPLE MISCLASSIFIED EXAMPLES =====")
    display_df = mis[["message", "actual_label", "pred_label"]].head(n)
    print(display_df.to_string(index=False))
    mis.head(n).to_csv(os.path.join(out_dir, "misclassified_samples.csv"), index=False)


def save_model(model, vectorizer, out_dir="models"):
    ensure_dir(out_dir)
    model_path = os.path.join(out_dir, "logreg_spam_model.joblib")
    vec_path = os.path.join(out_dir, "tfidf_vectorizer.joblib")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Saved model to {model_path} and vectorizer to {vec_path}")


def main():
    out_dir = ensure_dir("plots")
    models_dir = ensure_dir("models")

    df = load_data("spam.csv")
    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    X, y, vectorizer, df = prepare_data(df)

    X_train, X_test, y_train, y_test, msg_train, msg_test = train_test_split(
        X, y, df["cleaned"], test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    print("\n===== ACCURACY =====")
    print(accuracy_score(y_test, y_pred))
    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))

    # Additional metrics
    auc = roc_auc_score(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    print(f"\nROC AUC: {auc:.4f} — Average Precision (AP): {ap:.4f}")

    # Visualizations
    plot_class_distribution(df, out_dir)
    plot_confusion_matrix(y_test, y_pred, out_dir=out_dir)
    plot_roc_pr_curves(y_test, y_scores, out_dir=out_dir)
    plot_top_predictive_words(vectorizer, model, top_n=20, out_dir=out_dir)
    plot_wordclouds(df, out_dir=out_dir)
    show_misclassified_examples(msg_test, y_test, y_pred, out_dir=out_dir, n=10)
    save_model(model, vectorizer, out_dir=models_dir)

    print("\n===== TASK COMPLETED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()
# -----------------------------------------------------------
# TASK 4 - SPAM EMAIL CLASSIFICATION (OIBSIP)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import string
import re

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")   # change file name if needed

# some datasets have extra unnamed columns → drop them
df = df.iloc[:, :2]
df.columns = ['label', 'message']

print("\n===== FIRST 5 ROWS =====")
print(df.head())

# -----------------------------------------------------------
# TEXT CLEANING FUNCTION
# -----------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"\d+", "", text)       # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = text.strip()
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# convert labels to 0/1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------------------------------------------
# SPLIT FEATURES & TARGET
# -----------------------------------------------------------
X = df['cleaned_message']
y = df['label_num']

# -----------------------------------------------------------
# TF-IDF VECTORIZE
# -----------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(X)

# -----------------------------------------------------------
# TRAIN-TEST SPLIT
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# TRAIN MODEL
# -----------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------------
# PREDICT & EVALUATE
# -----------------------------------------------------------
y_pred = model.predict(X_test)

print("\n===== ACCURACY =====")
print(accuracy_score(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n===== TASK 4 COMPLETED SUCCESSFULLY =====")

# -----------------------------------------------------------
# EXAMPLE PREDICTION
# -----------------------------------------------------------
sample_email = "Congratulations! You won a free lottery. Call now to claim!"
sample_email = clean_text(sample_email)
sample_vector = vectorizer.transform([sample_email])
prediction = model.predict(sample_vector)

print("\nSample Email Prediction:", "SPAM" if prediction[0] == 1 else "HAM")
