"""
Create a small test dataset for quick training validation from real seizure labels.
This extracts a stratified subset of the full CHB-MIT dataset for testing.

IMPORTANT: This script requires real seizure labels (y values > 0).
If no seizures are found, it will fail with a clear error.
"""

import numpy as np
import sys
import argparse
from pathlib import Path


def create_test_dataset(
    input_dir: str = "D:/epimind/ml/data/chbmit_processed",
    output_dir: str = "D:/epimind/ml/data/chbmit_test",
    target_samples: int = 500,
):
    """
    Create test dataset from full CHB-MIT processed files with real stratified labels.
    
    Ensures:
    - Uses REAL seizure labels from preprocessed data
    - Stratified sampling: maintains class balance
    - Clear failure if no positive examples found
    
    Args:
        input_dir: Directory containing full NPZ files
        output_dir: Directory to save test files
        target_samples: Target total samples per file (balanced if possible)
        
    Raises:
        ValueError: If no positive windows found (real seizure labels missing)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npz_files = sorted(input_dir.glob("*.npz"))
    
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {input_dir}")
    
    print(f"Creating test dataset from {len(npz_files)} files")
    print(f"Target samples per file: {target_samples} (stratified)\n")
    
    total_positives = 0
    
    for npz_file in npz_files:
        print(f"Processing: {npz_file.name}")
        
        # Load full data
        data = np.load(npz_file, mmap_mode='r')
        X_full = np.array(data['X'], dtype=np.float32)
        y_full = np.array(data['y'], dtype=np.float32)
        
        print(f"  Full shape: X={X_full.shape}, y={y_full.shape}")
        
        # Get indices for each class
        pos_indices = np.where(y_full > 0.5)[0]
        neg_indices = np.where(y_full <= 0.5)[0]
        
        print(f"  Label distribution: {len(pos_indices):6d} positive, {len(neg_indices):6d} negative")
        
        # CRITICAL CHECK: Fail if no positive windows found
        if len(pos_indices) == 0:
            print(f"\n✗ ERROR: No positive windows found in {npz_file.name}")
            print(f"This means seizure labels were not properly extracted during preprocessing.")
            print(f"Run: python label_audit.py to verify seizure data exists in summary files.")
            print(f"Then re-run preprocessing: python build_chbmit_windows.py")
            raise ValueError(
                f"No positive windows in {npz_file.name}. "
                f"Check seizure label extraction in preprocessing."
            )
        
        total_positives += len(pos_indices)
        
        # Stratified sampling: try 50/50 split
        samples_per_class = target_samples // 2
        
        # Sample with stratification
        if len(pos_indices) >= samples_per_class:
            pos_sample = np.random.choice(pos_indices, samples_per_class, replace=False)
        else:
            # Not enough positives - use all and adjust negatives
            pos_sample = pos_indices
            samples_per_class = len(pos_indices)
        
        if len(neg_indices) >= samples_per_class:
            neg_sample = np.random.choice(neg_indices, samples_per_class, replace=False)
        else:
            # Not enough negatives - use all
            neg_sample = neg_indices
        
        # Combine indices
        indices = np.concatenate([pos_sample, neg_sample])
        
        X_test = X_full[indices]
        y_test = np.concatenate([
            np.ones(len(pos_sample), dtype=np.float32),
            np.zeros(len(neg_sample), dtype=np.float32)
        ])
        
        # Shuffle together
        shuffle_order = np.random.permutation(len(y_test))
        X_test = X_test[shuffle_order]
        y_test = y_test[shuffle_order]
        
        # Print stratification
        actual_pos = (y_test > 0.5).sum()
        actual_neg = (y_test <= 0.5).sum()
        print(f"  Test set: {actual_pos:6d} positive, {actual_neg:6d} negative ({actual_pos/(actual_pos+actual_neg)*100:.1f}%)")
        
        # Save test data
        output_path = output_dir / npz_file.name
        np.savez_compressed(
            output_path,
            X=X_test,
            y=y_test,
            fs=256.0,
        )
        file_size_mb = output_path.stat().st_size / (1024**2)
        print(f"  Saved: {output_path.name} ({file_size_mb:.1f} MB)\n")
    
    # Final summary
    print(f"{'=' * 80}")
    if total_positives == 0:
        print(f"✗ FAILED: No positive windows found across all patients")
        print(f"Seizure labels were not properly extracted.")
        raise ValueError("No positive windows found. Check preprocessing seizure label extraction.")
    else:
        print(f"✓ Test dataset created successfully")
        print(f"  Total positives found: {total_positives}")
        print(f"  Output directory: {output_dir}")
    
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create test dataset with real seizure labels from preprocessed data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="D:/epimind/ml/data/chbmit_processed",
        help="Input directory with preprocessed NPZ files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="D:/epimind/ml/data/chbmit_test",
        help="Output directory for test data"
    )
    parser.add_argument(
        "--per-patient",
        type=int,
        default=500,
        help="Target samples per patient (default: 500)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic labels (50/50 split) instead of real labels - NOT RECOMMENDED"
    )
    
    args = parser.parse_args()
    
    if args.synthetic:
        print("[WARNING] Using synthetic labels mode. This is NOT recommended for production.")
    else:
        print("[INFO] Using real seizure labels from preprocessing")
    
    try:
        create_test_dataset(
            input_dir=args.input,
            output_dir=args.output,
            target_samples=args.per_patient
        )
    except ValueError as e:
        print(f"\n✗ {e}")
        sys.exit(1)
