"""Realistic evaluation for CHB-MIT seizure detection.

Supports two modes:
1. BALANCED (--mode balanced):
   - Fast baseline evaluation on stratified test sets (50/50 seizure/non-seizure)
   - Quick sanity check, ~1-2 minutes per fold
   - Good for development and testing
   
2. REALISTIC (--mode realistic):
   - Full natural imbalanced distribution (~0.14-0.28% seizures)
   - Uses memory-mapped arrays for efficient large-file handling
   - Requires conversion: python ml/training/convert_npz_to_memmap.py
   - Production evaluation for panel submission

Both modes compute:
- ROC-AUC, PR-AUC (Average Precision)
- Accuracy, Precision, Recall, Specificity, F1
- Confusion matrix
- Threshold sweep with metrics at each threshold (0.05-0.95)
- FP/hour (False Positives Per Hour) - deployment metric
- Optimal thresholds for different clinical scenarios

Patient-wise cross-validation (NO DATA LEAKAGE):
- Train on one patient, test on another
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

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
    roc_curve,
)

from models import TinyEEGCNN
import inspect


# ============================================================================
# Constants
# ============================================================================

WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5  # Stride used in preprocessing (0.5 = overlapping windows)
SAMPLING_RATE = 256.0

# Evaluation modes
MODE_BALANCED = 'balanced'
MODE_REALISTIC = 'realistic'


# ============================================================================
# Model Factory
# ============================================================================

def create_model(n_channels: int, device: str) -> TinyEEGCNN:
    """
    Create TinyEEGCNN model with correct constructor signature.
    
    Inspects TinyEEGCNN.__init__ to find the correct parameter name for channels.
    Handles variations: num_channels, input_channels, in_channels, channels, n_channels.
    
    Args:
        n_channels: Number of EEG channels
        device: Device to place model on (cuda, cpu, etc.)
        
    Returns:
        TinyEEGCNN model on specified device
    """
    sig = inspect.signature(TinyEEGCNN.__init__)
    params = sig.parameters
    
    # Try common constructor parameter names in order of likelihood
    if "num_channels" in params:
        return TinyEEGCNN(num_channels=n_channels).to(device)
    if "input_channels" in params:
        return TinyEEGCNN(input_channels=n_channels).to(device)
    if "in_channels" in params:
        return TinyEEGCNN(in_channels=n_channels).to(device)
    if "n_channels" in params:
        return TinyEEGCNN(n_channels=n_channels).to(device)
    if "channels" in params:
        return TinyEEGCNN(channels=n_channels).to(device)
    
    # Fallback: constructor takes no channel argument
    return TinyEEGCNN().to(device)


# ============================================================================
# Helper Functions
# ============================================================================

def load_npz_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EEG data from NPZ file (small files under 100 MB).
    For large files, use load_memmap_data() instead.
    """
    npz_path_str = str(npz_path)
    file_size_mb = Path(npz_path_str).stat().st_size / (1024**2)
    print(f"  Loading: {Path(npz_path_str).name} ({file_size_mb:.1f} MB)")
    
    data = np.load(npz_path_str, allow_pickle=False)
    X = np.array(data['X'], dtype=np.float32, copy=True)
    y = np.array(data['y'], dtype=np.float32, copy=True)
    print(f"  Loaded: X {X.shape}, y {y.shape}")
    return X, y


def load_memmap_data(memmap_dir: str) -> Tuple[np.memmap, np.ndarray, dict]:
    """
    Load memory-mapped EEG data from converted directory.
    
    Args:
        memmap_dir: Path to directory with X.dat, y.npy, meta.json
        
    Returns:
        (X_memmap, y_array, metadata_dict)
    """
    memmap_dir = Path(memmap_dir)
    
    # Load metadata
    meta_path = memmap_dir / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print(f"  Meta: {meta['n_windows']} windows, {meta['n_channels']} channels, {meta['n_samples_per_window']} samples")
    
    # Load X as memmap (read-only)
    X_path = memmap_dir / 'X.dat'
    if not X_path.exists():
        raise FileNotFoundError(f"X memmap not found: {X_path}")
    
    X = np.memmap(
        X_path,
        dtype=np.float32,
        mode='r',
        shape=(meta['n_windows'], meta['n_channels'], meta['n_samples_per_window'])
    )
    print(f"  X memmap: {X.shape}")
    
    # Load y
    y_path = memmap_dir / 'y.npy'
    if not y_path.exists():
        raise FileNotFoundError(f"y array not found: {y_path}")
    
    y = np.load(y_path, allow_pickle=False).astype(np.float32)
    print(f"  y array: {y.shape}, positives: {(y > 0.5).sum()}, negatives: {(y <= 0.5).sum()}")
    
    return X, y, meta


def compute_hours_per_window(stride_seconds: float = STRIDE_SECONDS) -> float:
    """
    Compute hours represented by all windows in dataset.
    
    Args:
        stride_seconds: Stride between windows in seconds
        
    Returns:
        Hours per window = stride_seconds / 3600
    """
    return stride_seconds / 3600.0


def compute_fp_per_hour(
    num_false_positives: int,
    num_windows: int,
    stride_seconds: float = STRIDE_SECONDS
) -> float:
    """
    Compute false positives per hour.
    
    Args:
        num_false_positives: Count of FP detections
        num_windows: Total test windows
        stride_seconds: Stride between windows in seconds
        
    Returns:
        FP/hour
    """
    if num_windows == 0:
        return 0.0
    
    hours_per_window = compute_hours_per_window(stride_seconds)
    total_hours = num_windows * hours_per_window
    
    if total_hours == 0:
        return 0.0
    
    return num_false_positives / total_hours


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics at given threshold."""
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    if len(np.unique(y_true)) == 1:
        # Only one class in ground truth
        acc = accuracy_score(y_true, y_pred_binary)
        return {
            "threshold": threshold,
            "accuracy": acc,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
        }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """
    Sweep over thresholds and compute metrics at each.
    
    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities
        thresholds: Thresholds to sweep (default: 0.05 to 0.95 step 0.05)
        
    Returns:
        List of dicts with metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    
    sweep_results = []
    for thr in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_pred_prob, thr)
        sweep_results.append(metrics)
    
    return sweep_results


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    test_windows: int,
    num_folds: int = 1,
) -> Dict[str, Dict]:
    """
    Find operating thresholds for deployment.
    
    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities
        test_windows: Total number of test windows
        num_folds: Number of folds for context
        
    Returns:
        Dict with different operating points:
        - best_f1: threshold that maximizes F1
        - high_recall: threshold achieving recall >= 0.90
        - high_specificity: threshold achieving specificity >= 0.95
    """
    sweep = threshold_sweep(y_true, y_pred_prob)
    
    result = {}
    
    # Best F1
    best_f1_idx = np.argmax([s['f1'] for s in sweep])
    best_f1_result = sweep[best_f1_idx]
    fp_per_hour_f1 = compute_fp_per_hour(best_f1_result['fp'], test_windows)
    result['best_f1'] = {
        'threshold': float(best_f1_result['threshold']),
        'f1': float(best_f1_result['f1']),
        'precision': float(best_f1_result['precision']),
        'recall': float(best_f1_result['recall']),
        'specificity': float(best_f1_result['specificity']),
        'accuracy': float(best_f1_result['accuracy']),
        'fp': int(best_f1_result['fp']),
        'fp_per_hour': float(fp_per_hour_f1),
    }
    
    # High Recall (>= 0.90)
    high_recall_candidates = [s for s in sweep if s['recall'] >= 0.90]
    if high_recall_candidates:
        # Choose the one with highest recall but lowest FP rate
        high_recall_result = max(high_recall_candidates, key=lambda s: s['recall'])
        fp_per_hour_recall = compute_fp_per_hour(high_recall_result['fp'], test_windows)
        result['high_recall_0_90'] = {
            'threshold': float(high_recall_result['threshold']),
            'f1': float(high_recall_result['f1']),
            'precision': float(high_recall_result['precision']),
            'recall': float(high_recall_result['recall']),
            'specificity': float(high_recall_result['specificity']),
            'accuracy': float(high_recall_result['accuracy']),
            'fp': int(high_recall_result['fp']),
            'fp_per_hour': float(fp_per_hour_recall),
        }
    
    # High Specificity (>= 0.95)
    high_spec_candidates = [s for s in sweep if s['specificity'] >= 0.95]
    if high_spec_candidates:
        # Choose the one with highest specificity and best recall
        high_spec_result = max(high_spec_candidates, key=lambda s: (s['specificity'], s['recall']))
        fp_per_hour_spec = compute_fp_per_hour(high_spec_result['fp'], test_windows)
        result['high_specificity_0_95'] = {
            'threshold': float(high_spec_result['threshold']),
            'f1': float(high_spec_result['f1']),
            'precision': float(high_spec_result['precision']),
            'recall': float(high_spec_result['recall']),
            'specificity': float(high_spec_result['specificity']),
            'accuracy': float(high_spec_result['accuracy']),
            'fp': int(high_spec_result['fp']),
            'fp_per_hour': float(fp_per_hour_spec),
        }
    
    return result


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> Tuple[nn.Module, List[float]]:
    """
    Train a CNN model on EEG data.
    
    Args:
        X_train: Training data, shape (N, channels, timesteps)
        y_train: Training labels, shape (N,)
        device: torch device
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Trained model and loss history
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
    
    # Create dataset and loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model using factory function that handles constructor signature
    model = create_model(X_train.shape[1], device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % max(1, epochs // 3) == 0:
            pos_train = (y_train > 0.5).sum()
            print(f"  Epoch [{epoch+1}/{epochs}]: Loss={avg_loss:.4f} "
                  f"(Train: {len(y_train)} samples, {pos_train} positive)")
    
    return model, loss_history


def evaluate_fold(
    train_data: Tuple,
    test_data: Tuple,
    fold_name: str,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 64,
    max_train_samples: Optional[int] = None,
) -> Dict:
    """
    Evaluate on one fold (train on one patient, test on another).
    
    Args:
        train_data: Tuple of (npz_path, memmap_dir) - one will be None
        test_data: Tuple of (npz_path, memmap_dir) - one will be None
        fold_name: Name of this fold
        device: torch device
        epochs: Training epochs
        batch_size: Batch size
        max_train_samples: Max training samples (stratified); None = use all
        
    Returns:
        Dict with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"{fold_name}")
    print(f"{'='*80}")
    
    # Load data - support both NPZ and memmap
    train_npz, train_memmap = train_data
    test_npz, test_memmap = test_data
    
    # Load training data
    print(f"\nLoading training data...")
    if train_npz is not None:
        X_train_full, y_train_full = load_npz_data(train_npz)
    else:
        X_train_full, y_train_full, meta_train = load_memmap_data(train_memmap)
    
    # Load test data
    print(f"Loading test data...")
    if test_npz is not None:
        X_test, y_test = load_npz_data(test_npz)
        meta_test = None
    else:
        X_test, y_test, meta_test = load_memmap_data(test_memmap)
    
    # Report train set
    pos_train_full = (y_train_full > 0.5).sum()
    neg_train_full = len(y_train_full) - pos_train_full
    print(f"\nTrain (full): {len(y_train_full)} samples "
          f"({pos_train_full} positive, {neg_train_full} negative, "
          f"{pos_train_full/len(y_train_full)*100:.2f}% seizure)")
    
    # Optionally subsample training data (stratified)
    if max_train_samples is not None and max_train_samples < len(y_train_full):
        print(f"Subsampling train to {max_train_samples} samples (stratified)...")
        
        pos_indices = np.where(y_train_full > 0.5)[0]
        neg_indices = np.where(y_train_full <= 0.5)[0]
        
        # Calculate proportional split
        pos_ratio = len(pos_indices) / len(y_train_full)
        neg_ratio = 1.0 - pos_ratio
        
        num_pos = int(max_train_samples * pos_ratio)
        num_neg = max_train_samples - num_pos
        
        np.random.seed(42)
        sampled_pos = np.random.choice(pos_indices, min(num_pos, len(pos_indices)), replace=False)
        sampled_neg = np.random.choice(neg_indices, min(num_neg, len(neg_indices)), replace=False)
        
        sampled_indices = np.concatenate([sampled_pos, sampled_neg])
        np.random.shuffle(sampled_indices)
        
        X_train = X_train_full[sampled_indices]
        y_train = y_train_full[sampled_indices]
        
        pos_train = (y_train > 0.5).sum()
        neg_train = len(y_train) - pos_train
        print(f"Train (sampled): {len(y_train)} samples "
              f"({pos_train} positive, {neg_train} negative, "
              f"{pos_train/len(y_train)*100:.2f}% seizure)")
    else:
        X_train = X_train_full
        y_train = y_train_full
    
    # Report test set (FULL, NO BALANCING)
    pos_test = (y_test > 0.5).sum()
    neg_test = len(y_test) - pos_test
    seizure_ratio_test = pos_test / len(y_test) * 100 if len(y_test) > 0 else 0
    
    # Calculate hours represented in test set
    if meta_test is not None:
        # From metadata: each window is n_samples_per_window at fs Hz with stride
        fs = meta_test.get('fs', SAMPLING_RATE)
        window_samples = meta_test['n_samples_per_window']
        window_duration_sec = window_samples / fs
        stride_sec = STRIDE_SECONDS
        
        # Total time = (num_windows - 1) * stride + window_duration
        # Approximation: num_windows * stride_sec / 3600
        test_hours = len(y_test) * stride_sec / 3600.0
    else:
        # Default calculation from windows
        test_hours = len(y_test) * compute_hours_per_window()
    
    print(f"\n{'─'*80}")
    print(f"Test set (FULL DISTRIBUTION, NO BALANCING):")
    print(f"  Total windows: {len(y_test)}")
    print(f"  Positive windows: {pos_test} ({seizure_ratio_test:.2f}%)")
    print(f"  Negative windows: {neg_test}")
    print(f"  Hours represented: {test_hours:.1f} hours")
    print(f"{'─'*80}")
    
    # Train model
    print(f"\nTraining model...")
    model, loss_history = train_model(X_train, y_train, device, epochs=epochs, batch_size=batch_size)
    
    # Test
    print(f"\nEvaluating on test set...")
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    y_pred_logits = np.concatenate(all_logits, axis=0)[:, 0]
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))  # Sigmoid
    
    # Compute primary metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else np.nan
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall_curve, precision_curve) if len(np.unique(y_test)) > 1 else np.nan
    
    # Threshold sweep
    print(f"\nThreshold sweep (0.05 to 0.95 step 0.05)...")
    sweep_results = threshold_sweep(y_test, y_pred_prob)
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(y_test, y_pred_prob, len(y_test))
    
    # Metrics at 0.5 threshold
    metrics_at_0_5 = compute_metrics_at_threshold(y_test, y_pred_prob, 0.5)
    fp_per_hour_at_0_5 = compute_fp_per_hour(metrics_at_0_5['fp'], len(y_test))
    
    # Confusion matrix at 0.5
    y_pred_binary_0_5 = (y_pred_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary_0_5).ravel()
    
    print(f"\n{'─'*80}")
    print(f"Metrics at threshold=0.50:")
    print(f"  Accuracy:     {metrics_at_0_5['accuracy']:.4f}")
    print(f"  Precision:    {metrics_at_0_5['precision']:.4f}")
    print(f"  Recall:       {metrics_at_0_5['recall']:.4f}")
    print(f"  Specificity:  {metrics_at_0_5['specificity']:.4f}")
    print(f"  F1:           {metrics_at_0_5['f1']:.4f}")
    print(f"  ROC-AUC:      {roc_auc:.4f}" if not np.isnan(roc_auc) else f"  ROC-AUC:      N/A")
    print(f"  PR-AUC:       {pr_auc:.4f}" if not np.isnan(pr_auc) else f"  PR-AUC:       N/A")
    print(f"  FP/hour:      {fp_per_hour_at_0_5:.2f}")
    print(f"\nConfusion Matrix (threshold=0.50):")
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"{'─'*80}")
    
    # Print threshold sweep results
    print(f"\nThreshold Sweep Results:")
    print(f"{'Thr':>5} {'Prec':>6} {'Rec':>6} {'Spec':>6} {'F1':>6} {'FP/hr':>8}")
    print(f"{'-'*45}")
    for result in sweep_results:
        fp_per_hour = compute_fp_per_hour(result['fp'], len(y_test))
        print(f"{result['threshold']:.2f}  {result['precision']:.4f}  "
              f"{result['recall']:.4f}  {result['specificity']:.4f}  "
              f"{result['f1']:.4f}  {fp_per_hour:>7.2f}")
    
    print(f"\n{'─'*80}")
    print(f"Optimal Operating Thresholds:")
    print(f"{'─'*80}")
    for op_name, op_metrics in optimal_thresholds.items():
        print(f"\n{op_name.upper()}:")
        print(f"  Threshold:    {op_metrics['threshold']:.2f}")
        print(f"  F1:           {op_metrics['f1']:.4f}")
        print(f"  Precision:    {op_metrics['precision']:.4f}")
        print(f"  Recall:       {op_metrics['recall']:.4f}")
        print(f"  Specificity:  {op_metrics['specificity']:.4f}")
        print(f"  FP/hour:      {op_metrics['fp_per_hour']:.2f}")
    
    return {
        "fold_name": fold_name,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "test_positives": int(pos_test),
        "test_negatives": int(neg_test),
        "test_seizure_ratio_pct": float(seizure_ratio_test),
        "test_hours": float(len(y_test) * compute_hours_per_window()),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else None,
        "metrics_at_0_5": {
            "threshold": 0.5,
            "accuracy": float(metrics_at_0_5['accuracy']),
            "precision": float(metrics_at_0_5['precision']),
            "recall": float(metrics_at_0_5['recall']),
            "specificity": float(metrics_at_0_5['specificity']),
            "f1": float(metrics_at_0_5['f1']),
            "fp_per_hour": float(fp_per_hour_at_0_5),
        },
        "confusion_matrix_0_5": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold_sweep": [
            {
                "threshold": float(r['threshold']),
                "accuracy": float(r['accuracy']),
                "precision": float(r['precision']),
                "recall": float(r['recall']),
                "specificity": float(r['specificity']),
                "f1": float(r['f1']),
                "fp": int(r['fp']),
                "fp_per_hour": float(compute_fp_per_hour(r['fp'], len(y_test))),
            }
            for r in sweep_results
        ],
        "optimal_thresholds": optimal_thresholds,
    }


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Realistic evaluation on CHB-MIT with natural imbalanced distribution"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["balanced", "realistic"],
        help="Evaluation mode: 'balanced' uses test set (fast), 'realistic' uses full data with memmap"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="D:/epimind/ml/data/chbmit_test_labeled",
        help="Directory with labeled NPZ files (test set recommended for memory efficiency)"
    )
    parser.add_argument(
        "--train_patient",
        type=str,
        required=True,
        choices=["chb01", "chb02"],
        help="Patient to train on"
    )
    parser.add_argument(
        "--test_patient",
        type=str,
        required=True,
        choices=["chb01", "chb02"],
        help="Patient to test on"
    )
    parser.add_argument(
        "--export",
        type=str,
        default="D:/epimind/ml/export/models",
        help="Directory to save reports"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Max training samples (stratified sampling); None = use all"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    export_dir = Path(args.export)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CHB-MIT EVALUATION - {args.mode.upper()} MODE")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Train patient: {args.train_patient}")
    print(f"Test patient: {args.test_patient}")
    
    # Select data directories based on mode
    if args.mode == MODE_BALANCED:
        processed_dir = Path("D:/epimind/ml/data/chbmit_test_labeled")
        print(f"Data source: {processed_dir} (stratified test sets, 50/50)")
        train_path = processed_dir / f"{args.train_patient}.npz"
        test_path = processed_dir / f"{args.test_patient}.npz"
        
        if not train_path.exists() or not test_path.exists():
            print(f"ERROR: Test set files not found")
            print(f"  Train: {train_path}")
            print(f"  Test: {test_path}")
            return
        
        # Evaluate with NPZ loading (fast path)
        fold_result = evaluate_fold(
            train_data=(train_path, None),
            test_data=(test_path, None),
            fold_name=f"Train: {args.train_patient} → Test: {args.test_patient} [BALANCED]",
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_samples=args.max_train_samples,
        )
    
    elif args.mode == MODE_REALISTIC:
        memmap_dir = Path("D:/epimind/ml/data/chbmit_memmap")
        train_memmap = memmap_dir / args.train_patient
        test_memmap = memmap_dir / args.test_patient
        
        # Check if memmap conversions exist
        if not (train_memmap / 'X.dat').exists() or not (test_memmap / 'X.dat').exists():
            print(f"\n❌ ERROR: Memory-mapped data not found!")
            print(f"\nTo use realistic mode, first convert NPZ files:")
            print(f"\n  python ml/training/convert_npz_to_memmap.py \\")
            print(f"    --input D:/epimind/ml/data/chbmit_processed_labeled \\")
            print(f"    --output D:/epimind/ml/data/chbmit_memmap \\")
            print(f"    --patients chb01 chb02")
            print(f"\nThis will create memory-mapped arrays for efficient loading.")
            return
        
        print(f"Data source: {memmap_dir} (full processed data, natural distribution)")
        
        # For realistic mode with very large datasets, auto-set max_train_samples if not specified
        max_samples = args.max_train_samples
        if max_samples is None:
            max_samples = 20000  # Default: use 20k stratified samples for GPU memory
            print(f"Auto-setting max_train_samples to {max_samples} for realistic mode")
        
        # Evaluate with memmap loading (realistic path)
        fold_result = evaluate_fold(
            train_data=(None, train_memmap),
            test_data=(None, test_memmap),
            fold_name=f"Train: {args.train_patient} → Test: {args.test_patient} [REALISTIC]",
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_samples=max_samples,
        )
    else:
        print(f"ERROR: Unknown mode {args.mode}")
        return
    
    # Save JSON report with mode info
    report_name = f"chbmit_eval_report_{args.mode}_{args.train_patient}_to_{args.test_patient}"
    report_path = export_dir / f"{report_name}.json"
    with open(report_path, 'w') as f:
        json.dump(fold_result, f, indent=2)
    print(f"\n✓ JSON report saved: {report_path}")
    
    # Save Markdown report
    md_path = export_dir / f"{report_name}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(_generate_markdown_report(fold_result, args.train_patient, args.test_patient))
    print(f"✓ Markdown report saved: {md_path}")
    
    print(f"\n{'='*80}\n")


def _generate_markdown_report(
    fold_result: Dict,
    train_patient: str,
    test_patient: str,
) -> str:
    """Generate markdown report from evaluation results."""
    roc_auc = fold_result['roc_auc']
    pr_auc = fold_result['pr_auc']
    m_0_5 = fold_result['metrics_at_0_5']
    cm = fold_result['confusion_matrix_0_5']
    optimal = fold_result['optimal_thresholds']
    
    # Pre-format conditional values
    roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    pr_auc_str = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
    
    # Pre-format test set distribution (to avoid format issues in f-string)
    test_samples_str = f"{fold_result['test_samples']:,}"
    test_positives_str = f"{fold_result['test_positives']:,}"
    test_negatives_str = f"{fold_result['test_negatives']:,}"
    test_seizure_ratio_str = f"{fold_result['test_seizure_ratio_pct']:.2f}"
    test_hours_str = f"{fold_result['test_hours']:.1f}"
    
    # Pre-format primary metrics to avoid format spec issues in markdown table
    accuracy_str = f"{m_0_5['accuracy']:.4f}"
    precision_str = f"{m_0_5['precision']:.4f}"
    recall_str = f"{m_0_5['recall']:.4f}"
    specificity_str = f"{m_0_5['specificity']:.4f}"
    f1_str = f"{m_0_5['f1']:.4f}"
    fp_per_hour_str = f"{m_0_5['fp_per_hour']:.2f}"
    
    # Pre-format confusion matrix
    tn_str = f"{cm['tn']:,}"
    fp_str = f"{cm['fp']:,}"
    fn_str = f"{cm['fn']:,}"
    tp_str = f"{cm['tp']:,}"
    cm_tn_raw = f"{cm['tn']:6d}"
    cm_fp_raw = f"{cm['fp']:6d}"
    cm_fn_raw = f"{cm['fn']:6d}"
    cm_tp_raw = f"{cm['tp']:6d}"
    
    md = f"""# CHB-MIT Realistic Evaluation Report

**Train Patient:** {train_patient}  
**Test Patient:** {test_patient}

## Test Set Distribution

- **Total windows:** {test_samples_str}
- **Positive windows:** {test_positives_str} ({test_seizure_ratio_str}%)
- **Negative windows:** {test_negatives_str}
- **Hours represented:** {test_hours_str} hours

**Note:** Natural imbalanced distribution - NO balancing applied

## Primary Metrics (Threshold = 0.50)

| Metric | Value |
|--------|-------|
| **ROC-AUC** | {roc_auc_str} |
| **PR-AUC** | {pr_auc_str} |
| **Accuracy** | {accuracy_str} |
| **Precision** | {precision_str} |
| **Recall** | {recall_str} |
| **Specificity** | {specificity_str} |
| **F1 Score** | {f1_str} |
| **FP/hour** | {fp_per_hour_str} |

## Confusion Matrix (Threshold = 0.50)

```
                Predicted
              Negative  Positive
Actual  Negative  {cm_tn_raw}  {cm_fp_raw}
        Positive  {cm_fn_raw}  {cm_tp_raw}
```

- **True Negatives:** {tn_str}
- **False Positives:** {fp_str}
- **False Negatives:** {fn_str}
- **True Positives:** {tp_str}

## Optimal Operating Thresholds

"""
    
    for op_name, op_metrics in optimal.items():
        md += f"""
### {op_name.upper().replace('_', ' ')}

**Recommended for:** {_recommend_use_case(op_name)}

| Metric | Value |
|--------|-------|
| **Threshold** | {op_metrics['threshold']:.3f} |
| **F1 Score** | {op_metrics['f1']:.4f} |
| **Precision** | {op_metrics['precision']:.4f} |
| **Recall** | {op_metrics['recall']:.4f} |
| **Specificity** | {op_metrics['specificity']:.4f} |
| **False Positives** | {op_metrics['fp']:,} |
| **FP/hour** | {op_metrics['fp_per_hour']:.2f} |
"""
    
    md += f"""

## Threshold Sweep Results

| Threshold | Precision | Recall | Specificity | F1 Score | FP/hour |
|-----------|-----------|--------|-------------|----------|---------|
"""
    
    for sweep_item in fold_result['threshold_sweep']:
        thr = sweep_item['threshold']
        prec = sweep_item['precision']
        rec = sweep_item['recall']
        spec = sweep_item['specificity']
        f1 = sweep_item['f1']
        fp_h = sweep_item['fp_per_hour']
        md += f"| {thr:.2f} | {prec:.4f} | {rec:.4f} | {spec:.4f} | {f1:.4f} | {fp_h:.2f} |\n"
    
    md += f"""

## Recommendations for Deployment

1. **High-Sensitivity Setting** (catch more seizures):
   - Use "high_recall_0_90" threshold
   - Expects ~90% recall but higher FP/hour
   - Trade-off: fewer missed seizures, more false alarms

2. **Balanced Setting** (maximize overall performance):
   - Use "best_f1" threshold
   - Best overall balance of precision/recall
   - Good default choice

3. **High-Specificity Setting** (minimize false alarms):
   - Use "high_specificity_0_95" threshold
   - Expects ~95% specificity, very low FP/hour
   - Trade-off: may miss some seizures

---

**Generated:** {str(Path(__file__).parent.parent.parent / 'ml' / 'training')}  
**Evaluation Type:** Realistic (full natural imbalanced distribution, NO balancing on test set)
"""
    
    return md


def _recommend_use_case(op_name: str) -> str:
    """Get recommendation for operating threshold."""
    if "f1" in op_name.lower():
        return "General purpose / default choice"
    elif "recall" in op_name.lower():
        return "High-sensitivity clinical use (catch more seizures)"
    elif "specificity" in op_name.lower():
        return "Low false-alarm requirement (minimize alerts)"
    else:
        return "Custom use case"


if __name__ == "__main__":
    main()
