import os
import sys

# Thêm thư mục hiện tại vào sys.path để import module local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import prepare_data          # Hàm chuẩn bị dữ liệu
from models import MBTIClassifierML    # Classifier ML

def main():
    # ------------------------------
    # Chuẩn bị dữ liệu
    # ------------------------------
    # Trả về X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = prepare_data("data/mbti_1.csv")

    # ------------------------------
    # Khởi tạo và train model
    # ------------------------------
    model = MBTIClassifierML(use_svm=False)  # Dùng LogisticRegression (không SVM)
    model.fit(X_train, y_train)              # Train model trên tập huấn luyện

    # ------------------------------
    # Lưu model
    # ------------------------------
    os.makedirs("reports", exist_ok=True)   # Tạo thư mục nếu chưa tồn tại
    model.save("reports/mbti_ml.pkl")       # Lưu model
    print("✅ Saved ML model to reports/mbti_ml.pkl")

if __name__ == "__main__":
    main()
