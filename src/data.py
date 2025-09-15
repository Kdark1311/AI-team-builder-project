import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        posts = str(row["posts"]).replace("|||", "///").split("///")
        for p in posts:
            p = p.strip()
            if len(p) > 0:
                rows.append({
                    "post": p,
                    "mbti_IE": row["mbti_IE"],
                    "mbti_NS": row["mbti_NS"],
                    "mbti_TF": row["mbti_TF"],
                    "mbti_JP": row["mbti_JP"],
                })
    return pd.DataFrame(rows)

# ======================
# 4. Dataset class cho BERT
# ======================
class MBTIDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, augment_fn=None):
        self.df = df
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
def plot_distribution(df, col="type"):
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"Distribution of {col}")
    plt.show()

# ======================
# 6. Chia dữ liệu 70/20/10 và lưu ra file
# ======================
def split_and_save(df, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)

    # train 70%, temp 30%
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df[["mbti_IE","mbti_NS","mbti_TF","mbti_JP"]]
    )

    # valid 20%, test 10% (tỉ lệ trong temp: 2/3 và 1/3)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.3333,
        random_state=42,
        stratify=temp_df[["mbti_IE","mbti_NS","mbti_TF","mbti_JP"]]
    )

    print("Train:", len(train_df), "Valid:", len(valid_df), "Test:", len(test_df))

    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(save_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    return train_df, valid_df, test_df

if __name__ == "__main__":
    # 1. Load dữ liệu gốc
    df = load_data("D:\Progamming\Progamming_courses\Quorsk\project\data\mbti_1.csv")

    # 2. Thêm cột nhãn binary
    df = add_binary_columns(df)

    # 3. Tách các post ra từng dòng
    df_expanded = explode_posts(df)

    # 4. Vẽ phân phối nhãn MBTI
    plot_distribution(df, col="type")
    plot_distribution(df_expanded, col="mbti_IE")

    # 5. Chia và lưu ra train/valid/test
    train_df, valid_df, test_df = split_and_save(df_expanded, save_dir="data")

    print("✅ Đã lưu xong train.csv, valid.csv, test.csv trong thư mục /data/")
