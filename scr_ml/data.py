import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

# --- Set seed để tái lập kết quả ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

# --- Load CSV, in info cơ bản ---
def load_data(path):
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    print(f"Số mẫu: {len(df)}, Số loại MBTI: {df['type'].nunique()}")
    print(df["type"].value_counts())
    return df

# --- Chuyển MBTI string sang nhãn nhị phân 0/1 ---
def mbti_to_binary(mbti):
    return {
        "IE": 0 if mbti[0] == "I" else 1,
        "NS": 0 if mbti[1] == "N" else 1,
        "TF": 0 if mbti[2] == "T" else 1,
        "JP": 0 if mbti[3] == "J" else 1,
    }

# --- Thêm 4 cột nhị phân vào dataframe ---
def add_binary_columns(df):
    df = df.copy()
    for axis in ["IE","NS","TF","JP"]:
        df[f"mbti_{axis}"] = df["type"].apply(lambda x: mbti_to_binary(x)[axis])
    return df

# --- Chuẩn bị train/test split ---
def prepare_data(path, seed=42, test_size=0.2):
    set_seed(seed)  # set seed
    df = load_data(path)  # load CSV
    df = add_binary_columns(df)  # thêm nhãn nhị phân

    # Chuẩn hóa cột text
    if "posts" in df.columns:
        df["text"] = df["posts"].fillna("").astype(str).str.replace("///", " ").str.replace("|||", " ")
    else:
        df["text"] = df["post"].fillna("").astype(str)

    X = df["text"].values  # lấy cột text
    y = df[["mbti_IE","mbti_NS","mbti_TF","mbti_JP"]].values  # lấy nhãn nhị phân

    # Train/test split, stratify theo nhãn để giữ tỉ lệ
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
