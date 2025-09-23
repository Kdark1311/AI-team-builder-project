import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def load_data(path):
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    print("Số mẫu:", len(df))
    print("Số loại MBTI:", df["type"].nunique())
    print(df["type"].value_counts())
    return df

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

def prepare_data(path, seed=42, test_size=0.2):
    set_seed(seed)
    df = load_data(path)
    df = add_binary_columns(df)
    if "posts" in df.columns:
        df["text"] = df["posts"].fillna("").astype(str).str.replace("///", " ").str.replace("|||", " ")
    else:
        df["text"] = df["post"].fillna("").astype(str)
    X = df["text"].values
    y = df[["mbti_IE","mbti_NS","mbti_TF","mbti_JP"]].values
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
