# MBTI Personality Predictor

Ứng dụng dự đoán **MBTI** (16 loại tính cách) từ văn bản, sử dụng cả **Machine Learning truyền thống** và **BERT**.  
Ngoài việc phân loại, hệ thống còn gợi ý **vai trò phù hợp trong team** cho từng loại MBTI.

---

## 📂 Cấu trúc dự án

```
project/
│── data/                         # Dữ liệu gốc & tiền xử lý
│   ├── mbti_1.csv
│   ├── mbti_clean.csv
│   └── test.csv
│
│── demo/                         # Demo ứng dụng
│   ├── backend/                  # Flask backend (BERT model)
│   │   └── app.py
│   ├── frontend/                 # Giao diện web
│   │   ├── index.html
│   │   ├── main.js
│   │   └── style.css
│   └── UI_ML/                    # Streamlit UI (ML model)
│       └── app_ml.py
│
│── notebooks/                    # Notebook huấn luyện
│   ├── BERT_MBTI.ipynb
│   └── ML_MBTI.ipynb
│
│── reports/                      # Kết quả huấn luyện & model đã lưu
│   ├── BERT_report/
│   │   ├── classification_report.csv
│   │   ├── error_samples.csv
│   │   ├── metrics.csv
│   │   └── mbti_best.pt
│   └── ML_report/
│       ├── classification_reports_ml.txt
│       ├── metrics_ml.png
│       └── mbti_ml.pkl
│
│── scr_ml/                       # Code cho ML model
│   ├── data.py
│   ├── eval.py
│   ├── models.py
│   └── train.py
│
│── src/                          # Code cho BERT model
│   ├── data.py
│   ├── eval.py
│   ├── models.py
│   └── train.py
│
│── requirements.txt              # Thư viện cần thiết
│── README.md
└── .gitignore
```

---

## ⚙️ Cài đặt môi trường

```bash
git clone <repo-url>
cd project

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Cài dependencies
pip install -r requirements.txt
```

---

## 🚀 Chạy ứng dụng

### 1. Backend + Frontend (BERT + Flask)

```bash
python demo/backend/app.py
```

Mở trình duyệt tại: [http://127.0.0.1:5000](http://127.0.0.1:5000)  
👉 Nhập văn bản, hệ thống sẽ trả về MBTI + vai trò + giải thích.

### 2. UI Streamlit (ML model)

```bash
streamlit run demo/UI_ML/app_ml.py
```

Ứng dụng chạy trên [http://localhost:8501](http://localhost:8501).

---

## 📊 Kết quả huấn luyện

- **BERT**: lưu trong `reports/BERT_report/`
- **ML (SVM/Logistic/RandomForest, …)**: lưu trong `reports/ML_report/`

Ví dụ kết quả:

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| BERT  | 0.84     | 0.78     |
| ML    | 0.74     | 0.70     |

*(Chi tiết xem trong reports/)*

---

## 🛠 Công nghệ sử dụng

- **Ngôn ngữ**: Python
- **ML/DL**: scikit-learn, PyTorch, Transformers (HuggingFace)
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML, CSS, JS
- **Backend**: Flask, Streamlit

---

## 📌 Hướng phát triển

- Tích hợp thêm **API RESTful** cho ML model.  
- Nâng cấp giao diện frontend (React/Vue).  
- Triển khai lên **Docker + Render/Heroku**.  
- Thu thập thêm dữ liệu để cải thiện kết quả.  

---

## ☁️ Huấn luyện trên Kaggle

Do dung lượng dữ liệu và yêu cầu GPU, quá trình huấn luyện BERT được thực hiện trên **Kaggle Notebook**.

### 📓 Các notebook chính
- `ML_MBTI.ipynb` → Huấn luyện các mô hình ML (SVM, Logistic Regression, RandomForest, …).  
- `BERT_MBTI.ipynb` → Huấn luyện mô hình BERT với HuggingFace Transformers.  

⚠️ **Lưu ý:** Notebook `BERT_MBTI.ipynb` cần chạy trên Kaggle với **GPU (T4/P100)** và input chính là tập dữ liệu `mbti-type` (`data/mbti_1.csv` hoặc `data/mbti_clean.csv`).  

### 🚀 Cách chạy trên Kaggle
1. Đăng nhập [Kaggle](https://www.kaggle.com/).  
2. Vào mục **Code → New Notebook**.  
3. Upload 2 notebook (`ML_MBTI.ipynb`, `BERT_MBTI.ipynb`) cùng dataset `mbti-type`.  
4. Bật **GPU T4/P100** trong `Settings → Accelerator`.  
5. Chạy toàn bộ notebook.  
6. Model sau khi train sẽ được lưu vào thư mục `reports/` để sử dụng cho app.  
