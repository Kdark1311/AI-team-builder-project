from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib

class MBTIClassifierML:
    def __init__(self, use_svm=False, max_features=20000):
        """
        Khởi tạo ML MBTI classifier
        - use_svm: nếu True thì dùng LinearSVC, ngược lại LogisticRegression
        - max_features: số lượng feature tối đa cho TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, 
            ngram_range=(1,2), 
            stop_words="english"  # bỏ stopwords tiếng Anh
        )
        self.classifiers = {}  # dict lưu classifier cho từng axis
        self.use_svm = use_svm

    def fit(self, X_train, y_train):
        """
        Train model trên tập X_train, y_train
        X_train: list/array text
        y_train: numpy array (num_samples, 4) cho các axis IE, NS, TF, JP
        """
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        for i, axis in enumerate(["IE","NS","TF","JP"]):
            # Chọn classifier theo use_svm
            if self.use_svm:
                clf = LinearSVC(class_weight="balanced")
            else:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_train_tfidf, y_train[:, i])
            self.classifiers[axis] = clf

    def predict_text(self, text):
        """
        Dự đoán MBTI string từ một đoạn văn bản
        Trả về string 4 ký tự, ví dụ 'INTJ'
        """
        X_tfidf = self.vectorizer.transform([text])
        axes = ["IE","NS","TF","JP"]
        mbti = ""
        for axis in axes:
            mbti += self.classifiers[axis].predict(X_tfidf)[0]
        return mbti

    def save(self, path="ml_model.pkl"):
        """
        Lưu toàn bộ object MBTIClassifierML vào file .pkl
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path="ml_model.pkl"):
        """
        Load object MBTIClassifierML đã lưu từ file .pkl
        """
        return joblib.load(path)
