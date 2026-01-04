"""
Training pipeline for CHB-MIT seizure detection model.

Implements:
- Data loading from preprocessed .npz files
- Train/val split
- BCE with logits loss
- Accuracy and AUC metrics
- Model checkpoint saving
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from chbmit_dataset import load_chbmit_data
from models import TinyEEGCNN, TinyEEGResNet


def compute_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred_logits: Raw model logits
        
    Returns:
        Dictionary of metrics
    """
    # Convert logits to probabilities
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "auc": roc_auc_score(y_true, y_pred_prob),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
    }
    
    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Validation loop.
    
    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.eval()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item()
            
            # Collect predictions
            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(logits.cpu().numpy())
    
    # Compute metrics
    y_true = np.concatenate(all_y_true).flatten()
    y_pred = np.concatenate(all_y_pred).flatten()
    
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(y_true, y_pred)
    
    return avg_loss, metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.train()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
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
        
        # Collect predictions
        all_y_true.append(y_batch.cpu().detach().numpy())
        all_y_pred.append(logits.cpu().detach().numpy())
    
    # Compute metrics
    y_true = np.concatenate(all_y_true).flatten()
    y_pred = np.concatenate(all_y_pred).flatten()
    
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(y_true, y_pred)
    
    return avg_loss, metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    device: torch.device = None,
    checkpoint_dir: Path = None,
) -> Dict:
    """
    Full training loop.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to run on
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model.to(device)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_auc": [],
        "val_auc": [],
    }
    
    best_val_auc = 0.0
    best_epoch = 0
    
    print("\n" + "=" * 80)
    print(f"Training on device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_auc"].append(val_metrics["auc"])
        
        # Print log
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc:  {train_metrics['accuracy']:.4f} | Val Acc:  {val_metrics['accuracy']:.4f}")
        print(f"  Train AUC:  {train_metrics['auc']:.4f} | Val AUC:  {val_metrics['auc']:.4f}")
        print(f"  Train Prec: {train_metrics['precision']:.4f} | Val Prec: {val_metrics['precision']:.4f}")
        print(f"  Train Rec:  {train_metrics['recall']:.4f} | Val Rec:  {val_metrics['recall']:.4f}")
        print(f"  Train F1:   {train_metrics['f1']:.4f} | Val F1:   {val_metrics['f1']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            
            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_metrics["auc"],
                    "val_acc": val_metrics["accuracy"],
                }, checkpoint_path)
                print(f"  ★ New best model saved (AUC: {best_val_auc:.4f})")
    
    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"Best epoch: {best_epoch} (Val AUC: {best_val_auc:.4f})")
    print("=" * 80)
    
    return history


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train CHB-MIT seizure detection model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="D:/epimind/ml/data/chbmit_processed",
        help="Path to preprocessed CHB-MIT data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="D:/epimind/ml/export/models",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tiny_cnn", "tiny_resnet"],
        default="tiny_cnn",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.3,
        help="Fraction of full dataset to load (1.0 = all, 0.1 = 10%)",
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 80}")
    print(f"CHB-MIT Seizure Detection Training")
    print(f"{'=' * 80}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Sample ratio: {args.sample_ratio}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n{'=' * 80}")
    print("Loading dataset...")
    print(f"{'=' * 80}")
    train_loader, val_loader = load_chbmit_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_split=args.train_split,
        sample_ratio=args.sample_ratio,
    )
    
    # Create model
    print(f"\n{'=' * 80}")
    print("Creating model...")
    print(f"{'=' * 80}")
    
    if args.model_type == "tiny_cnn":
        model = TinyEEGCNN(num_channels=23, num_timesteps=512, dropout=0.3)
    elif args.model_type == "tiny_resnet":
        model = TinyEEGResNet(num_channels=23, num_timesteps=512, dropout=0.2)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Model: {args.model_type}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Train model
    checkpoint_dir = output_dir / "checkpoints"
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Save final model
    final_model_path = output_dir / f"chbmit_{args.model_type}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_type": args.model_type,
        "num_channels": 23,
        "num_timesteps": 512,
        "timestamp": datetime.now().isoformat(),
    }, final_model_path)
    
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Save training history
    history_path = output_dir / f"chbmit_{args.model_type}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Training history saved to: {history_path}")
    
    # Save training config
    config = {
        "model_type": args.model_type,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_split": args.train_split,
        "data_dir": args.data_dir,
        "num_parameters": model.count_parameters(),
        "timestamp": datetime.now().isoformat(),
    }
    
    config_path = output_dir / f"chbmit_{args.model_type}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Training config saved to: {config_path}")
    print(f"\n{'=' * 80}")
    print("Training pipeline complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
