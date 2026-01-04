# ml/training/train_model.py

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from .models import SimpleEEGNet
from .datasets import SyntheticEEGDataset


ROOT_DIR = Path(__file__).resolve().parents[2]  # D:/epimind
EXPORT_DIR = ROOT_DIR / "ml" / "export" / "models"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = EXPORT_DIR / "epimind_eeg_cnn.pt"
MODEL_VERSION = "eeg-cnn-v1"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    dataset = SyntheticEEGDataset(
        n_samples=8000,
        n_channels=4,
        window_size=512,
        seizure_prob=0.3,
    )

    # 80/20 train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = SimpleEEGNet(n_channels=4, n_samples=512).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 8
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)  # (B, C, T)
            y = y.to(device)  # (B, 1)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)

                preds = (outputs >= 0.5).float()
                correct += (preds == y).sum().item()
                total += y.numel()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"[Epoch {epoch}/{n_epochs}] "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"Val acc: {val_acc:.3f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "version": MODEL_VERSION,
                    "n_channels": 4,
                    "n_samples": 512,
                },
                CHECKPOINT_PATH,
            )
            print(f"[TRAIN] Saved best model to {CHECKPOINT_PATH}")

    print("[TRAIN] Done.")


if __name__ == "__main__":
    train()
