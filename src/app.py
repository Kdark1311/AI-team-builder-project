# data.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import numpy as np

# ====== 1. Đọc dữ liệu ======
df = pd.read_csv("mbti_1.csv")  # đổi path nếu cần

# ====== 2. Làm sạch văn bản ======
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)   # bỏ link
    text = text.replace("|||", " ")              # thay separator = khoảng trắng
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # giữ chữ cái và space
    return text.lower()

df["clean_posts"] = df["posts"].apply(clean_text)

# ====== 3. Chia nhãn MBTI thành 4 nhãn binary ======
df["E_I"] = df["type"].apply(lambda x: 1 if x[0] == "E" else 0)
df["S_N"] = df["type"].apply(lambda x: 1 if x[1] == "S" else 0)
df["T_F"] = df["type"].apply(lambda x: 1 if x[2] == "T" else 0)
df["J_P"] = df["type"].apply(lambda x: 1 if x[3] == "J" else 0)

# ====== 4. Tokenizer ======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encodings = tokenizer(
    list(df["clean_posts"]),
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="np",
)

# ====== 5. Đánh giá tỷ lệ bị truncate ======
lengths = [len(tokenizer.tokenize(t)) for t in df["clean_posts"]]
too_long = sum(l > 512 for l in lengths)
print(f"Tổng số mẫu: {len(df)}")
print(f"Số mẫu dài hơn 512 token: {too_long} ({too_long/len(df)*100:.2f}%)")

# ====== 6. Chia dữ liệu ======
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["type"]
)
train_df, valid_df = train_test_split(
    train_df, test_size=0.1, random_state=42, stratify=train_df["type"]
)

print("Train:", len(train_df), "Valid:", len(valid_df), "Test:", len(test_df))

# ====== 7. Lưu ra file riêng ======
train_df.to_csv("train_binary.csv", index=False)
valid_df.to_csv("valid_binary.csv", index=False)
test_df.to_csv("test_binary.csv", index=False)

print("Đã lưu train/valid/test CSV với nhãn binary 4 cột.")
