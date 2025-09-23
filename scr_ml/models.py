from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib

class MBTIClassifierML:
    def __init__(self, use_svm=False, max_features=20000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words="english")
        self.classifiers = {}
        self.use_svm = use_svm

    def fit(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        for i, axis in enumerate(["IE","NS","TF","JP"]):
            if self.use_svm:
                clf = LinearSVC(class_weight="balanced")
            else:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_train_tfidf, y_train[:, i])
            self.classifiers[axis] = clf

    def predict_text(self, text):
        """Dự đoán MBTI string từ một đoạn văn bản"""
        X_tfidf = self.vectorizer.transform([text])
        axes = ["IE","NS","TF","JP"]
        mbti = ""
        for axis in axes:
            mbti += self.classifiers[axis].predict(X_tfidf)[0]
        return mbti

    def save(self, path="ml_model.pkl"):
        """Lưu toàn bộ object"""
        joblib.dump(self, path)

    @staticmethod
    def load(path="ml_model.pkl"):
        """Load object đã lưu"""
        return joblib.load(path)
