# src/data.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import matplotlib.pyplot as plt

# ======================
# 1. Load & inspect data
# ======================
def load_data(path):
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    print("Số mẫu:", len(df))
    print("Số loại MBTI khác nhau:", df["type"].nunique())
    print(df["type"].value_counts())
    return df

# ======================
# 2. Encode MBTI -> 4 nhãn binary
# ======================
def mbti_to_binary(mbti):
    return {
        "IE": 0 if mbti[0] == "I" else 1,
        "NS": 0 if mbti[1] == "N" else 1,
        "TF": 0 if mbti[2] == "T" else 1,
        "JP": 0 if mbti[3] == "J" else 1,
    }

def add_binary_columns(df):
    df = df.copy()
    df["mbti_IE"] = df["type"].apply(lambda x: mbti_to_binary(x)["IE"])
    df["mbti_NS"] = df["type"].apply(lambda x: mbti_to_binary(x)["NS"])
    df["mbti_TF"] = df["type"].apply(lambda x: mbti_to_binary(x)["TF"])
    df["mbti_JP"] = df["type"].apply(lambda x: mbti_to_binary(x)["JP"])
    return df

# ======================
# 3. Làm sạch văn bản
# ======================
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)             # bỏ link
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)          # bỏ ký tự lạ
    text = re.sub(r"\s+", " ", text).strip()                  # gom khoảng trắng
    return text

# ======================
# 4. Dataset class cho BERT
# ======================
class MBTIDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, augment_fn=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = clean_text(row.get("posts", ""))  # 👈 giữ nguyên toàn bộ posts của user

        if self.augment_fn:
            try:
                text = self.augment_fn(text)
            except:
                pass

        labels = torch.tensor(
            [row["mbti_IE"], row["mbti_NS"], row["mbti_TF"], row["mbti_JP"]],
            dtype=torch.float
        )

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels
        }

# ======================
# 5. Chạy trực tiếp để tiền xử lý & lưu
# ======================
if __name__ == "__main__":
    df = load_data("/kaggle/input/mbti-type/mbti_1.csv")
    df = add_binary_columns(df)
    df["posts"] = df["posts"].apply(clean_text)   # làm sạch toàn bộ posts (không explode)
    df.to_csv("/kaggle/working/mbti_clean.csv", index=False)
    print("✅ Đã lưu xong mbti_clean.csv (giữ nguyên 1 dòng/user, có 4 nhãn binary)")
    
    
    label_cols = ["mbti_IE", "mbti_NS", "mbti_TF", "mbti_JP"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, col in enumerate(label_cols):
        counts = df[col].value_counts().sort_index()  # 0 và 1
        counts.plot(kind="bar", ax=axes[i])
        axes[i].set_title(f"Phân phối nhãn {col}")
        axes[i].set_xticklabels(["0", "1"], rotation=0)
        for idx, val in enumerate(counts):
            axes[i].text(idx, val, str(val), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()
