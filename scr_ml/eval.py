import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os 
import sys 
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import prepare_data
from models import MBTIClassifierML

def main():
    X_train, X_test, y_train, y_test = prepare_data("data/mbti_1.csv")

    # Load model
    model = MBTIClassifierML.load(
        r"D:\Progamming\Progamming_courses\Quorsk\project\reports\mbti_ml.pkl"
    )

    # Dự đoán từng axis
    preds = []
    for axis, clf in model.classifiers.items():
        preds.append(clf.predict(model.vectorizer.transform(X_test)))
    preds = np.array(preds).T  # shape = (num_samples, 4)

    axes = ["IE", "NS", "TF", "JP"]
    accs, f1s = [], []

    # File để lưu classification reports
    report_path = r"D:\Progamming\Progamming_courses\Quorsk\project\reports\classification_reports.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        for i, axis in enumerate(axes):
            acc = accuracy_score(y_test[:, i], preds[:, i])
            f1 = f1_score(y_test[:, i], preds[:, i], average="macro")
            accs.append(acc)
            f1s.append(f1)

            # In ra console
            print(f"{axis} - Acc: {acc:.4f}, F1: {f1:.4f}")
            print(classification_report(y_test[:, i], preds[:, i]))

            # Ghi vào file txt
            f.write(f"\n===== {axis} =====\n")
            f.write(f"Acc: {acc:.4f}, F1: {f1:.4f}\n")
            f.write(classification_report(y_test[:, i], preds[:, i]))
            f.write("\n\n")

    print(f"✅ Classification reports saved at {report_path}")

    # Vẽ biểu đồ Accuracy & F1 cho 4 axis
    x = np.arange(len(axes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, accs, width, label="Accuracy")
    ax.bar(x + width/2, f1s, width, label="F1 Score")

    ax.set_xticks(x)
    ax.set_xticklabels(axes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("MBTI Axis Classification Performance")
    ax.legend()

    save_path = r"D:\Progamming\Progamming_courses\Quorsk\project\reports\metrics.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Metrics chart saved at {save_path}")

if __name__ == "__main__":
    main()
