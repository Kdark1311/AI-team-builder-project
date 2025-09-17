import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import re
def set_seed(seed=42):
    import random, numpy as _np, torch as _torch
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

# ======================
# 1. Load & inspect data
# ======================
def load_data(path):
    df = pd.read_csv(path)
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
# 3. Tách posts theo dấu /// hoặc |||
# ======================
def explode_posts(df):
    rows = []
    for _, row in df.iterrows():
        posts = str(row.get("posts", "")).replace("|||", "///").split("///")
        for p in posts:
            p = clean_text(p)   # 👈 thêm bước làm sạch
            if len(p) > 0:
                rows.append({
                    "post": p,
                    "mbti_IE": row["mbti_IE"],
                    "mbti_NS": row["mbti_NS"],
                    "mbti_TF": row["mbti_TF"],
                    "mbti_JP": row["mbti_JP"],
                })
    return pd.DataFrame(rows)
#làm sạch văn bản
def clean_text(text: str) -> str:
    text = str(text)

    # 1. Bỏ link (http, https, www)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 2. Bỏ emoji dạng :smile: hoặc :blushed:
    text = re.sub(r":\w+:", " ", text)

    # 3. Bỏ ký tự không phải chữ, số, dấu câu cơ bản
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)

    # 4. Gom nhiều khoảng trắng thành 1
    text = re.sub(r"\s+", " ", text).strip()

    return text
# ======================
# 6. Dataset class cho BERT
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
        text = row["post"]

        if self.augment_fn:
            text = self.augment_fn(text)

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
# 5. Visualization helpers
# ======================
def plot_distribution(df, col="type", save_path=None):
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"Distribution of {col}")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# ======================
# 6. Chia dữ liệu 70/20/10 theo USER rồi mới explode
# ======================
def split_and_save(df, save_dir="data", seed=42):
    os.makedirs(save_dir, exist_ok=True)

    # ⚠️ split theo user-level (mỗi dòng = 1 user)
    train_users, temp_users = train_test_split(
        df,
        test_size=0.3,
        random_state=seed,
        stratify=df["type"]  # stratify theo type gốc
    )

    valid_users, test_users = train_test_split(
        temp_users,
        test_size=0.3333,
        random_state=seed,
        stratify=temp_users["type"]
    )

    # Sau khi chia user → explode thành post-level
    train_df = explode_posts(train_users)
    valid_df = explode_posts(valid_users)
    test_df  = explode_posts(test_users)

    print("Users -> Train:", len(train_users), "Valid:", len(valid_users), "Test:", len(test_users))
    print("Posts -> Train:", len(train_df), "Valid:", len(valid_df), "Test:", len(test_df))

    # Lưu
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(save_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    return train_df, valid_df, test_df


if __name__ == "__main__":
    # ví dụ chạy nhanh (bạn có thể thay path của bạn)
    df = load_data("D:/Progamming/Progamming_courses/Quorsk/project/data/mbti_1.csv")
    df = add_binary_columns(df)

    # 👇 không explode ở đây, chỉ plot trên user-level
    plot_distribution(df, col="type")

    # Split theo user rồi mới explode
    train_df, valid_df, test_df = split_and_save(df, save_dir="data")

    # visualize phân phối sau khi explode
    plot_distribution(train_df, col="mbti_IE")
    print("✅ Đã lưu xong train.csv, valid.csv, test.csv trong thư mục /data/")

