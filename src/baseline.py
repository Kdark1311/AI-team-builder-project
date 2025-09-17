import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from data import set_seed

def run_baseline(data_dir="data", seed=42, use_svm=False, max_features=20000):
    set_seed(seed)

    # ✅ Đọc data đã xử lý từ data.py (post-level, đã làm sạch)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # Gộp train + valid để có nhiều dữ liệu hơn (80%)
    full_train = pd.concat([train_df, valid_df], axis=0)

    X_train, y_train = full_train["post"].values, full_train[["mbti_IE","mbti_NS","mbti_TF","mbti_JP"]].values
    X_test, y_test   = test_df["post"].values, test_df[["mbti_IE","mbti_NS","mbti_TF","mbti_JP"]].values

    # ✅ TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,2),
        stop_words="english"
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # ✅ Train 4 classifier (1 cho mỗi trục MBTI)
    preds = np.zeros_like(y_test)
    metrics = {}

    for i, axis in enumerate(["IE","NS","TF","JP"]):
        if use_svm:
            clf = LinearSVC()
        else:
            clf = LogisticRegression(max_iter=1000)

        clf.fit(X_train_tfidf, y_train[:, i])
        y_pred = clf.predict(X_test_tfidf)
        preds[:, i] = y_pred

        acc = accuracy_score(y_test[:, i], y_pred)
        f1  = f1_score(y_test[:, i], y_pred, average="macro")
        metrics[axis] = {"Accuracy": acc, "Macro-F1": f1}
        print(f"{axis}: Acc={acc:.4f}, F1={f1:.4f}")

    # ✅ Lưu metrics
    metrics_df = pd.DataFrame(metrics).T
    os.makedirs("reports", exist_ok=True)
    metrics_df.to_csv("reports/baseline_metrics.csv")
    print("\n✅ Baseline metrics saved to reports/baseline_metrics.csv")

    return metrics_df


if __name__ == "__main__":
    run_baseline()
