# ======================
# Evaluation
# ======================
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# ----------------------
# Core evaluation
# ----------------------
def evaluate(model, dataloader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_labels.append(labels)
            all_preds.append(preds)
            all_probs.append(probs)

    return (
        np.vstack(all_labels),
        np.vstack(all_preds),
        np.vstack(all_probs)
    )

# ----------------------
# Confusion matrices
# ----------------------
def plot_confusion_matrices(y_true, y_pred, axes, save_dir=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"Confusion Matrix - {axes[i]}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "confusion_matrices.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Saved confusion matrix to {path}")
    else:
        plt.show()

# ----------------------
# Probability distribution
# ----------------------
def plot_probability_distribution(probs, axes, save_dir=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        sns.histplot(probs[:, i], bins=20, kde=True, ax=ax)
        ax.set_title(f"Predicted Prob Distribution - {axes[i]}")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "prob_dist.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Saved probability distribution to {path}")
    else:
        plt.show()

# ----------------------
# Error analysis
# ----------------------
def error_analysis(test_df, y_true, y_pred, y_probs, axes, n_samples=15):
    errors = []
    for i in range(len(y_true)):
        for j, axis in enumerate(axes):
            if y_true[i, j] != y_pred[i, j]:
                errors.append({
                    "index": i,
                    "axis": axis,
                    "true": int(y_true[i, j]),
                    "pred": int(y_pred[i, j]),
                    "prob": float(y_probs[i, j]),
                    "text": str(test_df.iloc[i]["posts"])[:200] + "..."
                })

    errors_df = pd.DataFrame(errors)
    os.makedirs("reports", exist_ok=True)
    errors_df.to_csv("reports/error_samples.csv", index=False)
    print(f"\n❌ Saved {len(errors_df)} misclassified samples to reports/error_samples.csv")

    if len(errors_df) > 0:
        print("\n=== Sample Errors ===")
        print(errors_df.sample(min(n_samples, len(errors_df))))

# ----------------------
# Main
# ----------------------
def main():

    # Config
    set_seed(42)
    model_name = "bert-base-uncased"
    batch_size = 8
    max_len = 256
    model_path = "/kaggle/working/mbti_best.pt"
    test_csv = "/kaggle/working/test.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load test set
    test_df = pd.read_csv(test_csv)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    test_dataset = MBTIDataset(test_df, tokenizer, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    model = MBTIModel(model_name=model_name, pooling="cls+mean", dropout=0.4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluate
    y_true, y_pred, y_probs = evaluate(model, test_loader, device)

    axes = ["IE", "NS", "TF", "JP"]
    metrics = {}

    print("\n=== Metrics per axis ===")
    for i, axis in enumerate(axes):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i], average="macro")
        metrics[axis] = {"Accuracy": acc, "Macro-F1": f1}
        print(f"{axis}: Acc={acc:.4f}, Macro-F1={f1:.4f}")

    metrics_df = pd.DataFrame(metrics).T
    os.makedirs("reports", exist_ok=True)
    metrics_df.to_csv("reports/metrics.csv")
    print("\n✅ Metrics saved to reports/metrics.csv")

    # Save classification report (multi-label)
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).to_csv("reports/classification_report.csv")
    print("✅ Classification report saved to reports/classification_report.csv")

    # Visualization
    plot_confusion_matrices(y_true, y_pred, axes, save_dir="reports/figures")
    plot_probability_distribution(y_probs, axes, save_dir="reports/figures")

    # Error analysis
    error_analysis(test_df, y_true, y_pred, y_probs, axes)


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    main()
