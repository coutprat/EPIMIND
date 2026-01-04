"""
Patient-wise evaluation of CHB-MIT seizure detection model.

NO DATA LEAKAGE: Train on one patient, test on another.
- Fold 1: Train on chb01, test on chb02
- Fold 2: Train on chb02, test on chb01

Metrics computed:
- Accuracy, Precision, Recall, Specificity, F1, ROC-AUC
- Confusion matrix
- Best threshold optimization on train set
"""

import json
import numpy as np
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
    roc_curve,
)

from models import TinyEEGCNN


def load_npz_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EEG data and labels from NPZ file.
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        Tuple of (X, y) as numpy arrays
    """
    data = np.load(npz_path, mmap_mode='r')
    X = np.array(data['X'], dtype=np.float32)
    y = np.array(data['y'], dtype=np.float32)
    return X, y


def compute_metrics(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics at given threshold.
    
    Args:
        y_true: Ground truth labels
        y_pred_logits: Raw logits from model
        threshold: Decision threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Convert logits to probabilities
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    # Handle edge case: all predictions are same class
    if len(np.unique(y_true)) == 1:
        # Only one class in true labels
        acc = accuracy_score(y_true, y_pred_binary)
        return {
            "accuracy": acc,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "roc_auc": np.nan,
        }
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # ROC-AUC (if both classes present)
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


def find_best_threshold(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
) -> float:
    """
    Find threshold that maximizes F1 score on training set.
    
    Args:
        y_true: Ground truth labels
        y_pred_logits: Raw logits
        
    Returns:
        Optimal threshold value
    """
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    
    # Try thresholds from 0.1 to 0.9
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
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = None,
) -> float:
    """
    Train model on given dataset.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
        
    Returns:
        Final training loss
    """
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
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    final_loss = total_loss / len(train_loader)
    return final_loss


def evaluate_fold(
    train_npz: str,
    test_npz: str,
    fold_name: str,
    device: torch.device,
    batch_size: int = 128,
    epochs: int = 10,
) -> Dict:
    """
    Evaluate model on one fold.
    
    Args:
        train_npz: Path to training NPZ file
        test_npz: Path to test NPZ file
        fold_name: Name of fold (e.g., "Fold 1: chb01→chb02")
        device: Device to use
        batch_size: Batch size
        epochs: Training epochs
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'=' * 80}")
    print(f"{fold_name}")
    print(f"{'=' * 80}")
    
    # Load data
    print(f"Loading training data from {Path(train_npz).name}...", end=" ", flush=True)
    X_train, y_train = load_npz_data(train_npz)
    print(f"✓ {X_train.shape}")
    
    print(f"Loading test data from {Path(test_npz).name}...", end=" ", flush=True)
    X_test, y_test = load_npz_data(test_npz)
    print(f"✓ {X_test.shape}")
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create and train model
    print(f"\nTraining model for {epochs} epochs on {device}...", end=" ", flush=True)
    model = TinyEEGCNN(num_channels=23, num_timesteps=512, dropout=0.3)
    final_loss = train_model(model, train_loader, epochs=epochs, device=device)
    print(f"✓ (final loss: {final_loss:.4f})")
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        # Train predictions
        X_train_tensor = torch.from_numpy(X_train).to(device)
        train_logits = model(X_train_tensor).cpu().numpy().flatten()
        
        # Test predictions
        X_test_tensor = torch.from_numpy(X_test).to(device)
        test_logits = model(X_test_tensor).cpu().numpy().flatten()
    
    # Find best threshold on train set
    print(f"\nFinding optimal threshold on train set...", end=" ", flush=True)
    best_threshold = find_best_threshold(y_train, train_logits)
    print(f"✓ (threshold: {best_threshold:.3f})")
    
    # Evaluate at threshold 0.5
    print(f"\nMetrics at threshold = 0.5:")
    print(f"{'─' * 80}")
    
    metrics_0_5_train = compute_metrics(y_train, train_logits, threshold=0.5)
    metrics_0_5_test = compute_metrics(y_test, test_logits, threshold=0.5)
    
    print_metrics("Train", metrics_0_5_train)
    print_metrics("Test ", metrics_0_5_test)
    
    # Evaluate at best threshold
    print(f"\nMetrics at optimal threshold = {best_threshold:.3f}:")
    print(f"{'─' * 80}")
    
    metrics_best_train = compute_metrics(y_train, train_logits, threshold=best_threshold)
    metrics_best_test = compute_metrics(y_test, test_logits, threshold=best_threshold)
    
    print_metrics("Train", metrics_best_train)
    print_metrics("Test ", metrics_best_test)
    
    # Compute confusion matrix at best threshold
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
    
    # Return results
    return {
        "fold_name": fold_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_seizure_ratio": float(y_train.mean()),
        "test_seizure_ratio": float(y_test.mean()),
        "epochs": epochs,
        "best_threshold": float(best_threshold),
        "metrics_at_0_5": {
            "train": {k: (float(v) if not np.isnan(v) else None) for k, v in metrics_0_5_train.items()},
            "test": {k: (float(v) if not np.isnan(v) else None) for k, v in metrics_0_5_test.items()},
        },
        "metrics_at_best": {
            "train": {k: (float(v) if not np.isnan(v) else None) for k, v in metrics_best_train.items()},
            "test": {k: (float(v) if not np.isnan(v) else None) for k, v in metrics_best_test.items()},
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def print_metrics(label: str, metrics: Dict[str, float]):
    """Print metrics in formatted table."""
    print(f"  {label} | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | "
          f"Rec: {metrics['recall']:.4f} | Spec: {metrics['specificity']:.4f} | "
          f"F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f if not np.isnan(metrics['roc_auc']) else 'N/A'}")


def main():
    """Run patient-wise evaluation."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("D:/epimind/ml/data/chbmit_processed")
    export_dir = Path("D:/epimind/ml/export/models")
    
    print(f"\n{'=' * 80}")
    print(f"CHB-MIT Patient-wise Seizure Detection Evaluation")
    print(f"{'=' * 80}")
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    
    # Ensure data files exist
    chb01_path = data_dir / "chb01.npz"
    chb02_path = data_dir / "chb02.npz"
    
    if not chb01_path.exists():
        print(f"ERROR: {chb01_path} not found!")
        return
    if not chb02_path.exists():
        print(f"ERROR: {chb02_path} not found!")
        return
    
    # Evaluate both folds
    results = []
    
    # Fold 1: Train on chb01, test on chb02
    fold1_result = evaluate_fold(
        train_npz=str(chb01_path),
        test_npz=str(chb02_path),
        fold_name="Fold 1: Train on chb01 → Test on chb02",
        device=device,
        batch_size=128,
        epochs=10,
    )
    results.append(fold1_result)
    
    # Fold 2: Train on chb02, test on chb01
    fold2_result = evaluate_fold(
        train_npz=str(chb02_path),
        test_npz=str(chb01_path),
        fold_name="Fold 2: Train on chb02 → Test on chb01",
        device=device,
        batch_size=128,
        epochs=10,
    )
    results.append(fold2_result)
    
    # Compute averages
    print(f"\n{'=' * 80}")
    print(f"Cross-Patient Evaluation Summary")
    print(f"{'=' * 80}")
    
    avg_acc_0_5 = np.mean([r["metrics_at_0_5"]["test"]["accuracy"] for r in results])
    avg_f1_0_5 = np.mean([r["metrics_at_0_5"]["test"]["f1"] for r in results])
    avg_auc_0_5 = np.nanmean([r["metrics_at_0_5"]["test"]["roc_auc"] for r in results])
    
    avg_acc_best = np.mean([r["metrics_at_best"]["test"]["accuracy"] for r in results])
    avg_f1_best = np.mean([r["metrics_at_best"]["test"]["f1"] for r in results])
    avg_auc_best = np.nanmean([r["metrics_at_best"]["test"]["roc_auc"] for r in results])
    
    print(f"\nAt threshold = 0.5:")
    print(f"  Average Test Accuracy: {avg_acc_0_5:.4f}")
    print(f"  Average Test F1-score: {avg_f1_0_5:.4f}")
    print(f"  Average Test ROC-AUC:  {avg_auc_0_5:.4f}")
    
    print(f"\nAt optimal threshold:")
    print(f"  Average Test Accuracy: {avg_acc_best:.4f}")
    print(f"  Average Test F1-score: {avg_f1_best:.4f}")
    print(f"  Average Test ROC-AUC:  {avg_auc_best:.4f}")
    
    # Save results
    eval_report = {
        "evaluation": "Patient-wise Cross-Validation (NO DATA LEAKAGE)",
        "device": str(device),
        "folds": results,
        "summary": {
            "avg_test_accuracy_at_0_5": float(avg_acc_0_5),
            "avg_test_f1_at_0_5": float(avg_f1_0_5),
            "avg_test_auc_at_0_5": float(avg_auc_0_5),
            "avg_test_accuracy_at_best": float(avg_acc_best),
            "avg_test_f1_at_best": float(avg_f1_best),
            "avg_test_auc_at_best": float(avg_auc_best),
        },
    }
    
    report_path = export_dir / "chbmit_eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(eval_report, f, indent=2)
    
    print(f"\n✓ Evaluation report saved to: {report_path}")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
