"""
Baseline: TF-IDF + One-vs-Rest Logistic Regression (or LinearSVC)
Saves per-axis metrics and a combined report (accuracy + macro-F1).
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from data import load_data, add_binary_columns, set_seed

def run_baseline(data_path="D:\Progamming\Progamming_courses\Quorsk\project\data\mbti_1.csv", seed=42, use_svm=False, max_features=20000):
    set_seed(seed)
    df = load_data(data_path)
    df = add_binary_columns(df)
    # For baseline, we treat each row as a user blob: concatenate posts if "posts" exists
    if "posts" in df.columns:
        df["text"] = df["posts"].fillna("").astype(str).str.replace("///", " ").str.replace("|||", " ")
    elif "post" in df.columns:
        df["text"] = df["post"].fillna("").astype(str)
    else:
        raise ValueError("No posts/posts column found in data file.")

    X = df["text"].values
    y = df[["mbti_IE","mbti_NS","mbti_TF","mbti_JP"]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifiers = {}
    preds = np.zeros_like(y_test)
    probs = None

    for i, axis in enumerate(["IE","NS","TF","JP"]):
        if use_svm:
            clf = LinearSVC()
        else:
            # Use One-vs-Rest logistic for probabilistic output (approx via predict_proba)
            base = LogisticRegression(max_iter=1000)
            clf = OneVsRestClassifier(base) if False else base  # here single target so base is fine
        clf.fit(X_train_tfidf, y_train[:, i])
        y_pred = clf.predict(X_test_tfidf)
        preds[:, i] = y_pred
        classifiers[axis] = clf

    # Metrics
    metrics = {}
    for i, axis in enumerate(["IE","NS","TF","JP"]):
        acc = accuracy_score(y_test[:, i], preds[:, i])
        f1 = f1_score(y_test[:, i], preds[:, i], average='macro')
        metrics[axis] = {"Accuracy": float(acc), "Macro-F1": float(f1)}
    metrics_df = pd.DataFrame(metrics).T
    os.makedirs("reports", exist_ok=True)
    metrics_df.to_csv("reports/baseline_metrics.csv")
    print("Baseline metrics:\n", metrics_df)
    return metrics_df, classifiers, vectorizer

if __name__ == "__main__":
    run_baseline()
