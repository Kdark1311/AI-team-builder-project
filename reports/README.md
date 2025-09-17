# MBTI Personality Classification

Dự án phân loại MBTI từ văn bản (bài post) bằng baseline TF-IDF + Logistic Regression/SVM và mô hình nâng cao BERT.

---

## 📂 Cấu trúc thư mục

```
project/
  data/
    mbti.csv         # dữ liệu gốc
    train.csv        # train set (70%)
    valid.csv        # valid set (20%)
    test.csv         # test set (10%)
  notebooks/
    main.ipynb       # EDA + baseline thử nghiệm nhanh
  src/
    data.py          # xử lý dữ liệu, Dataset class cho BERT
    models.py        # định nghĩa MBTIModel (BERT classifier)
    train.py         # train BERT
    eval.py          # evaluate BERT, confusion matrix, error analysis
    baseline.py      # baseline TF-IDF + Logistic/SVM
    app.py           # (chưa viết) prototype Team Builder
  reports/
    metrics.csv, robustness.csv, error_samples.csv, figures/
  requirements.txt
  README.md
```

---

## ⚙️ Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu: Python >= 3.8, CUDA (nếu có GPU).

---

## 🗂️ Chuẩn bị dữ liệu

- Đặt file dữ liệu gốc `mbti.csv` vào thư mục `data/`.
- Format: mỗi dòng gồm ít nhất 2 cột: `type` (MBTI label, ví dụ INTJ) và `posts` (nội dung, phân tách bằng `///` hoặc `|||`).

---

## 🚀 Chạy baseline

```bash
python src/baseline.py
```

Kết quả baseline (Accuracy + Macro-F1 cho 4 trục MBTI) được lưu tại:

- `reports/baseline_metrics.csv`

---

## 🚀 Train BERT

```bash
python src/train.py
```

Mô hình tốt nhất được lưu tại:

- `checkpoints/mbti_model.pt`

---

## 📊 Đánh giá BERT

```bash
python src/eval.py
```

Kết quả bao gồm:

- `reports/metrics.csv`  
- `reports/error_samples.csv`  
- `reports/robustness.csv`  
- `reports/figures/` (confusion matrix, probability distribution)

---

## 📒 Notebook

- `notebooks/main.ipynb` có ví dụ EDA (phân phối type, độ dài post) và baseline Logistic Regression nhanh.

--- 
