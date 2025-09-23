import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import prepare_data
from models import MBTIClassifierML

def main():
    # Chuẩn bị dữ liệu
    X_train, X_test, y_train, y_test = prepare_data("data/mbti_1.csv")

    # Khởi tạo model
    model = MBTIClassifierML(use_svm=False)
    model.fit(X_train, y_train)

    # Lưu model
    os.makedirs("reports", exist_ok=True)
    model.save("reports/mbti_ml.pkl")
    print("✅ Saved ML model to reports/mbti_ml.pkl")

if __name__ == "__main__":
    main()
