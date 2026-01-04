"""
Quick evaluation script using test dataset.

This version runs on the small test dataset (1000 windows) for fast demonstration.
For full evaluation on complete CHB-MIT data, see evaluate_chbmit_full.py

Patient-wise evaluation (NO DATA LEAKAGE):
- Train on chb01, test on chb02
- Train on chb02, test on chb01
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from models import TinyEEGCNN


def load_npz_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load EEG data from NPZ file."""
    data = np.load(npz_path)
    X = np.array(data['X'], dtype=np.float32)
    y = np.array(data['y'], dtype=np.float32)
    return X, y


def compute_metrics(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics at given threshold."""
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    if len(np.unique(y_true)) == 1:
        acc = accuracy_score(y_true, y_pred_binary)
        return {
            "accuracy": acc,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "roc_auc": np.nan,
        }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    else:
        roc_auc = np.nan
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def find_best_threshold(y_true: np.ndarray, y_pred_logits: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 1.0, 0.05):
        y_pred_binary = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: torch.device = None,
) -> float:
    """Train model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_fold(
    train_npz: str,
    test_npz: str,
    fold_name: str,
    device: torch.device,
) -> Dict:
    """Evaluate model on one fold."""
    print(f"\n{'=' * 80}")
    print(f"{fold_name}")
    print(f"{'=' * 80}")
    
    print(f"Loading training data from {Path(train_npz).name}...", end=" ", flush=True)
    X_train, y_train = load_npz_data(train_npz)
    print(f"✓ {X_train.shape}")
    
    print(f"Loading test data from {Path(test_npz).name}...", end=" ", flush=True)
    X_test, y_test = load_npz_data(test_npz)
    print(f"✓ {X_test.shape}")
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print(f"Training model on {device}...", end=" ", flush=True)
    model = TinyEEGCNN(num_channels=23, num_timesteps=512, dropout=0.3)
    final_loss = train_model(model, train_loader, epochs=5, device=device)
    print(f"✓ (loss: {final_loss:.4f})")
    
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.from_numpy(X_train).to(device)
        train_logits = model(X_train_tensor).cpu().numpy().flatten()
        
        X_test_tensor = torch.from_numpy(X_test).to(device)
        test_logits = model(X_test_tensor).cpu().numpy().flatten()
    
    print(f"Finding optimal threshold...", end=" ", flush=True)
    best_threshold = find_best_threshold(y_train, train_logits)
    print(f"✓ (threshold: {best_threshold:.3f})")
    
    print(f"\nResults at threshold = 0.5:")
    print(f"{'─' * 80}")
    metrics_0_5_test = compute_metrics(y_test, test_logits, threshold=0.5)
    print_metrics("Test Acc", metrics_0_5_test)
    
    print(f"\nResults at optimal threshold = {best_threshold:.3f}:")
    print(f"{'─' * 80}")
    metrics_best_test = compute_metrics(y_test, test_logits, threshold=best_threshold)
    print_metrics("Test Acc", metrics_best_test)
    
    y_pred_prob = 1.0 / (1.0 + np.exp(-test_logits))
    y_pred_binary = (y_pred_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary, labels=[0, 1])
    
    # Handle confusion matrix safely
    if cm.size == 1:
        # Only one class predicted
        if y_test[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix (test set):")
    print(f"{'─' * 80}")
    print(f"  True Negatives:  {tn:6d}  |  False Positives: {fp:6d}")
    print(f"  False Negatives: {fn:6d}  |  True Positives:  {tp:6d}")
    
    return {
        "fold_name": fold_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_threshold": float(best_threshold),
        "metrics_at_0_5": {
            "test": {k: (float(v) if not np.isnan(v) else None) for k, v in metrics_0_5_test.items()},
        },
        "metrics_at_best": {
            "test": {k: (float(v) if not np.isnan(v) else None) for k, v in metrics_best_test.items()},
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_metrics(label: str, metrics: Dict[str, float]):
    """Print metrics."""
    auc_str = f"{metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else "N/A"
    print(f"  {label}: Acc={metrics['accuracy']:.4f} | Prec={metrics['precision']:.4f} | "
          f"Rec={metrics['recall']:.4f} | Spec={metrics['specificity']:.4f} | "
          f"F1={metrics['f1']:.4f} | AUC={auc_str}")


def main():
    """Run evaluation."""
    parser = argparse.ArgumentParser(
        description="Quick evaluation on CHB-MIT test dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="D:/epimind/ml/data/chbmit_test",
        help="Directory with test NPZ files"
    )
    parser.add_argument(
        "--export",
        type=str,
        default="D:/epimind/ml/export/models",
        help="Directory to save evaluation report"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data)
    export_dir = Path(args.export)
    
    print(f"\n{'=' * 80}")
    print(f"CHB-MIT Quick Evaluation (Test Dataset)")
    print(f"{'=' * 80}")
    print(f"Device: {device}")
    print(f"Data: {data_dir}")
    
    chb01_path = data_dir / "chb01.npz"
    chb02_path = data_dir / "chb02.npz"
    
    if not chb01_path.exists() or not chb02_path.exists():
        print("ERROR: Test dataset not found. Run create_test_dataset.py first.")
        return
    
    # Fold 1
    fold1_result = evaluate_fold(
        train_npz=str(chb01_path),
        test_npz=str(chb02_path),
        fold_name="Fold 1: Train on chb01 → Test on chb02",
        device=device,
    )
    
    # Fold 2
    fold2_result = evaluate_fold(
        train_npz=str(chb02_path),
        test_npz=str(chb01_path),
        fold_name="Fold 2: Train on chb02 → Test on chb01",
        device=device,
    )
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"Summary")
    print(f"{'=' * 80}")
    
    results = [fold1_result, fold2_result]
    avg_acc = np.mean([r["metrics_at_best"]["test"]["accuracy"] for r in results])
    avg_f1 = np.mean([r["metrics_at_best"]["test"]["f1"] for r in results])
    avg_auc = np.nanmean([r["metrics_at_best"]["test"]["roc_auc"] for r in results])
    
    print(f"Average Test Accuracy: {avg_acc:.4f}")
    print(f"Average Test F1-score: {avg_f1:.4f}")
    print(f"Average Test ROC-AUC:  {avg_auc:.4f}")
    
    # Save
    eval_report = {
        "evaluation": "Quick Evaluation (Test Dataset - NO DATA LEAKAGE)",
        "device": str(device),
        "data_dir": str(data_dir),
        "folds": results,
        "summary": {
            "avg_test_accuracy": float(avg_acc),
            "avg_test_f1": float(avg_f1),
            "avg_test_auc": float(avg_auc),
        },
    }
    
    report_path = export_dir / "chbmit_eval_report_quick.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(eval_report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
