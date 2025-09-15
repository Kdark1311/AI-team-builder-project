# src/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from data import MBTIDataset
from models import MBTIModel


def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # ======================
    # Config
    # ======================
    model_name = "bert-base-uncased"
    batch_size = 16
    max_len = 256
    lr = 2e-5
    epochs = 50
    patience = 5  # số epoch chờ cải thiện
    save_path = "mbti_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================
    # Load data
    # ======================
    from data import load_data, add_binary_columns, explode_posts, split_and_save

    df = load_data("data/mbti.csv")
    df = add_binary_columns(df)
    df = explode_posts(df)
    train_df, valid_df, _ = split_and_save(df, save_dir="data")

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = MBTIDataset(train_df, tokenizer, max_len=max_len)
    valid_dataset = MBTIDataset(valid_df, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # ======================
    # Model + Optimizer + Loss
    # ======================
    model = MBTIModel(model_name=model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = nn.BCEWithLogitsLoss()

    # ======================
    # Training loop with Early Stopping
    # ======================
    best_valid_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
        valid_loss = eval_epoch(model, valid_loader, device, loss_fn)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f}")

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered!")
                break


if __name__ == "__main__":
    main()
