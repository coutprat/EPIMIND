"""
CHB-MIT Dataset loader for seizure detection.
Loads preprocessed .npz files and returns PyTorch-compatible tensors.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List


class CHBMITDataset(Dataset):
    """
    Dataset class for CHB-MIT EEG data.
    
    Loads preprocessed .npz files containing:
    - X: (num_windows, num_channels, num_timesteps)
    - y: (num_windows,) binary labels (0=normal, 1=seizure)
    
    Uses mmap_mode='r' to reduce memory footprint during loading.
    """
    
    def __init__(self, data_dir: str = "D:/epimind/ml/data/chbmit_processed", sample_ratio: float = 1.0):
        """
        Initialize dataset by loading all .npz files from data_dir.
        
        Args:
            data_dir: Path to directory containing .npz files
            sample_ratio: Fraction of data to load (1.0 = all, 0.1 = 10%)
        """
        self.data_dir = Path(data_dir)
        self.sample_ratio = sample_ratio
        
        # Load all .npz files and concatenate
        all_X = []
        all_y = []
        
        npz_files = sorted(self.data_dir.glob("*.npz"))
        
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")
        
        print(f"Loading {len(npz_files)} .npz files from {self.data_dir}")
        
        for npz_file in npz_files:
            print(f"  → {npz_file.name}", end=" ", flush=True)
            # Use mmap_mode to reduce memory during loading
            data = np.load(npz_file, mmap_mode='r')
            X = np.array(data['X'], dtype=np.float32)
            y = np.array(data['y'], dtype=np.float32)
            
            # Sample if needed
            if sample_ratio < 1.0:
                n_samples = int(len(X) * sample_ratio)
                indices = np.random.choice(len(X), n_samples, replace=False)
                X = X[indices]
                y = y[indices]
            
            all_X.append(X)
            all_y.append(y)
            
            print(f"[X: {X.shape}, y: {y.shape}]", flush=True)
        
        # Concatenate all data
        self.X = np.concatenate(all_X, axis=0)
        self.y = np.concatenate(all_y, axis=0)
        
        print(f"\nDataset loaded successfully!")
        print(f"  Total samples: {len(self.X)}")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
        print(f"  Seizure ratio: {self.y.mean():.4f}")
        print(f"  Memory usage: {(self.X.nbytes + self.y.nbytes) / (1024**3):.2f} GB", flush=True)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (X, y) as torch tensors
            X: (channels, timesteps) float32
            y: scalar float32 for BCE
        """
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


def load_chbmit_data(
    data_dir: str = "D:/epimind/ml/data/chbmit_processed",
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 0,
    shuffle: bool = True,
    sample_ratio: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CHB-MIT dataset and create train/val dataloaders.
    
    Args:
        data_dir: Path to preprocessed data directory
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training (0.0-1.0)
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle training data
        sample_ratio: Fraction of full dataset to load (1.0 = all, 0.1 = 10%)
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Load full dataset
    dataset = CHBMITDataset(data_dir, sample_ratio=sample_ratio)
    
    # Create train/val split
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"\nDataLoader created:")
    print(f"  Train samples: {len(train_dataset)} batches")
    print(f"  Val samples: {len(val_dataset)} batches")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    train_loader, val_loader = load_chbmit_data(batch_size=16)
    
    # Inspect first batch
    for X_batch, y_batch in train_loader:
        print(f"\nFirst batch from train_loader:")
        print(f"  X batch shape: {X_batch.shape}")
        print(f"  y batch shape: {y_batch.shape}")
        print(f"  X dtype: {X_batch.dtype}")
        print(f"  y dtype: {y_batch.dtype}")
        print(f"  y values: {y_batch[:5]}")
        break
