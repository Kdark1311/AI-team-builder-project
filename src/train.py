import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import random 
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # deterministic cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 
# ======================
# Train + Eval helpers
# ======================
def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn, epoch, epochs, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device, loss_fn, epoch, epochs, mode="Valid"):
    model.eval()
    total_loss = 0
    all_labels, all_preds = [], []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [{mode}]", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            # Lấy xác suất + dự đoán nhị phân
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            # Lưu labels dưới dạng numpy
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)

            progress_bar.set_postfix(loss=loss.item())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(dataloader), acc, f1


def main():
    # Config
    model_name = "bert-base-uncased"
    batch_size = 8
    max_len = 256
    lr = 2e-5
    epochs = 50
    num_workers = 4
    save_dir = "/kaggle/working/"
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "mbti_best.pt")
    ckpt_path = os.path.join(save_dir, "mbti_ckpt.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================
    # Load & preprocess data
    # ======================
    data_dir = "/kaggle/working/"
    os.makedirs(data_dir, exist_ok=True)

    clean_csv = os.path.join(data_dir, "mbti_clean.csv")
    if not os.path.exists(clean_csv):
        df = load_data(os.path.join(data_dir, "mbti_1.csv"))
        df = add_binary_columns(df)
        df["posts"] = df["posts"].apply(clean_text)
        df.to_csv(clean_csv, index=False)

    df = pd.read_csv(clean_csv)

    # Split train/valid/test (70/20/10)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["type"]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.3333, random_state=42, stratify=temp_df["type"]
    )
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Tokenizer + Dataset + DataLoader
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = MBTIDataset(train_df, tokenizer, max_len=max_len)
    valid_dataset = MBTIDataset(valid_df, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)

    # Model + Optimizer + Loss
    model = MBTIModel(model_name=model_name, pooling="cls+mean", dropout=0.4).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # pos_weight cho BCE
    labels = train_df[["mbti_IE","mbti_NS","mbti_TF","mbti_JP"]].values
    pos_weights = (labels.shape[0] - labels.sum(axis=0)) / labels.sum(axis=0)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    scaler = GradScaler()

    # Resume checkpoint nếu có
    start_epoch, best_valid_f1 = 0, 0.0
    if os.path.exists(ckpt_path):
        print(f"🔄 Found checkpoint: {ckpt_path}, loading...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_valid_f1 = ckpt["best_valid_f1"]
        print(f"👉 Resume training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn, epoch, epochs, scaler)
        valid_loss, valid_acc, valid_f1 = eval_epoch(model, valid_loader, device, loss_fn, epoch, epochs)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | "
              f"Acc: {valid_acc:.4f} | Macro-F1: {valid_f1:.4f}")

        # Save best model (Macro-F1)
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model to {best_model_path}")

        # Luôn lưu checkpoint mỗi epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_valid_f1": best_valid_f1,
        }, ckpt_path)
        print(f"💾 Saved checkpoint (epoch {epoch + 1}) to {ckpt_path}")


if __name__ == "__main__":
    main()
