"""
Real evaluation script using true CHB-MIT seizure labels.

Patient-wise cross-validation (NO DATA LEAKAGE):
- Fold 1: Train on chb01, test on chb02
- Fold 2: Train on chb02, test on chb01

This script evaluates on REAL seizure labels extracted during preprocessing.
Metrics: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    auc,
    precision_recall_curve,
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
    """
    Compute comprehensive classification metrics at given threshold.
    
    Returns dict with:
    - accuracy, precision, recall, specificity, f1
    - roc_auc, pr_auc
    - tn, fp, fn, tp
    """
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    # Handle single-class case
    if len(np.unique(y_true)) == 1:
        accuracy = accuracy_score(y_true, y_pred_binary)
        return {
            "accuracy": float(accuracy),
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "pr_auc": None,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        roc_auc = None
    
    # PR-AUC
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(recall_vals, precision_vals)
    except ValueError:
        pr_auc = None
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_best_threshold(y_true: np.ndarray, y_pred_logits: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on training set."""
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 128,
) -> Tuple[nn.Module, np.ndarray]:
    """Train model on training data and return final logits."""
    
    # Create dataset
    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train).float()
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    # Get final training logits for threshold optimization
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_train).to(device)
        train_logits = model(X_tensor).cpu().numpy().squeeze()
    
    return model, train_logits


def evaluate_fold(
    train_npz: str,
    test_npz: str,
    fold_name: str,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 128,
) -> Dict:
    """
    Evaluate one fold: train on one patient, test on another.
    
    Returns dictionary with all metrics and confusion matrix.
    """
    print(f"\n{'=' * 80}")
    print(f"{fold_name}")
    print(f"{'=' * 80}")
    
    # Load data
    X_train, y_train = load_npz_data(train_npz)
    X_test, y_test = load_npz_data(test_npz)
    
    print(f"Train: {X_train.shape[0]:6d} samples ({(y_train > 0.5).sum():5d} positive, {(y_train <= 0.5).sum():5d} negative)")
    print(f"Test:  {X_test.shape[0]:6d} samples ({(y_test > 0.5).sum():5d} positive, {(y_test <= 0.5).sum():5d} negative)")
    
    # Train model
    model = TinyEEGCNN()
    model, train_logits = train_model(
        model, X_train, y_train, device,
        epochs=epochs,
        batch_size=batch_size
    )
    print(f"\nModel trained successfully")
    
    # Find best threshold on train set
    best_threshold = find_best_threshold(y_train, train_logits)
    print(f"Best threshold (F1-maximized on train): {best_threshold:.3f}")
    
    # Get test predictions
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).to(device)
        test_logits = model(X_test_tensor).cpu().numpy().squeeze()
    
    # Compute metrics at default threshold
    print(f"\n{'─' * 80}")
    print(f"Metrics at threshold = 0.500:")
    print(f"{'─' * 80}")
    metrics_test_05 = compute_metrics(y_test, test_logits, threshold=0.5)
    
    print(f"  Accuracy:   {metrics_test_05['accuracy']:.4f}")
    print(f"  Precision:  {metrics_test_05['precision']:.4f}")
    print(f"  Recall:     {metrics_test_05['recall']:.4f}")
    print(f"  Specificity:{metrics_test_05['specificity']:.4f}")
    print(f"  F1-Score:   {metrics_test_05['f1']:.4f}")
    if metrics_test_05['roc_auc'] is not None:
        print(f"  ROC-AUC:    {metrics_test_05['roc_auc']:.4f}")
    if metrics_test_05['pr_auc'] is not None:
        print(f"  PR-AUC:     {metrics_test_05['pr_auc']:.4f}")
    
    # Compute metrics at best threshold
    print(f"\n{'─' * 80}")
    print(f"Metrics at optimal threshold = {best_threshold:.3f}:")
    print(f"{'─' * 80}")
    metrics_test_best = compute_metrics(y_test, test_logits, threshold=best_threshold)
    
    print(f"  Accuracy:   {metrics_test_best['accuracy']:.4f}")
    print(f"  Precision:  {metrics_test_best['precision']:.4f}")
    print(f"  Recall:     {metrics_test_best['recall']:.4f}")
    print(f"  Specificity:{metrics_test_best['specificity']:.4f}")
    print(f"  F1-Score:   {metrics_test_best['f1']:.4f}")
    if metrics_test_best['roc_auc'] is not None:
        print(f"  ROC-AUC:    {metrics_test_best['roc_auc']:.4f}")
    if metrics_test_best['pr_auc'] is not None:
        print(f"  PR-AUC:     {metrics_test_best['pr_auc']:.4f}")
    
    # Confusion matrix at best threshold
    y_pred_prob = 1.0 / (1.0 + np.exp(-test_logits))
    y_pred_binary = (y_pred_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary, labels=[0, 1])
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    print(f"\nConfusion Matrix (at best threshold):")
    print(f"  True Negatives:  {int(tn):6d}  |  False Positives: {int(fp):6d}")
    print(f"  False Negatives: {int(fn):6d}  |  True Positives:  {int(tp):6d}")
    
    return {
        "fold_name": fold_name,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_positive_count": int((y_train > 0.5).sum()),
        "train_negative_count": int((y_train <= 0.5).sum()),
        "test_positive_count": int((y_test > 0.5).sum()),
        "test_negative_count": int((y_test <= 0.5).sum()),
        "best_threshold": float(best_threshold),
        "metrics_at_0_5": {
            "test": metrics_test_05
        },
        "metrics_at_best": {
            "test": metrics_test_best
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }


def main():
    """Run full evaluation on real seizure labels."""
    
    print("\n" + "=" * 80)
    print("CHB-MIT Full Evaluation (REAL SEIZURE LABELS)")
    print("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data paths
    data_dir = Path("D:/epimind/ml/data/chbmit_processed")
    chb01_path = data_dir / "chb01.npz"
    chb02_path = data_dir / "chb02.npz"
    
    if not chb01_path.exists() or not chb02_path.exists():
        print(f"ERROR: Processed data not found!")
        print(f"  chb01: {chb01_path.exists()}")
        print(f"  chb02: {chb02_path.exists()}")
        return
    
    print(f"Data: {data_dir}")
    
    # Run folds
    results = {
        "evaluation": "Full Evaluation (Real Seizure Labels)",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "data_dir": str(data_dir),
        "folds": [],
    }
    
    fold1_result = evaluate_fold(
        train_npz=str(chb01_path),
        test_npz=str(chb02_path),
        fold_name="Fold 1: Train on chb01 → Test on chb02",
        device=device,
        epochs=10,
        batch_size=128,
    )
    results["folds"].append(fold1_result)
    
    fold2_result = evaluate_fold(
        train_npz=str(chb02_path),
        test_npz=str(chb01_path),
        fold_name="Fold 2: Train on chb02 → Test on chb01",
        device=device,
        epochs=10,
        batch_size=128,
    )
    results["folds"].append(fold2_result)
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"Summary")
    print(f"{'=' * 80}")
    
    avg_acc = np.mean([r["metrics_at_best"]["test"]["accuracy"] for r in results["folds"]])
    avg_f1 = np.mean([r["metrics_at_best"]["test"]["f1"] for r in results["folds"]])
    avg_auc = np.mean([r["metrics_at_best"]["test"]["roc_auc"] for r in results["folds"] if r["metrics_at_best"]["test"]["roc_auc"] is not None])
    
    print(f"Average Test Accuracy: {avg_acc:.4f}")
    print(f"Average Test F1-score: {avg_f1:.4f}")
    if not np.isnan(avg_auc):
        print(f"Average Test ROC-AUC:  {avg_auc:.4f}")
    
    results["summary"] = {
        "avg_test_accuracy": float(avg_acc),
        "avg_test_f1": float(avg_f1),
        "avg_test_auc": float(avg_auc),
    }
    
    # Save report
    report_path = Path("D:/epimind/ml/export/models/chbmit_eval_report_real.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
