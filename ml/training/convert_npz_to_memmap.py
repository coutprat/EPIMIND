"""
Convert NPZ files to memory-mapped arrays for efficient large-file handling.

This enables realistic evaluation on full datasets without loading entire arrays into RAM.

Usage:
    python ml/training/convert_npz_to_memmap.py --input ml/data/chbmit_processed_labeled --output ml/data/chbmit_memmap --patients chb01 chb02
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys


def convert_npz_to_memmap(
    npz_path: str,
    output_dir: str,
    patient_name: str,
    verbose: bool = True,
) -> dict:
    """
    Convert single NPZ file to memory-mapped arrays.
    
    Args:
        npz_path: Path to input NPZ file
        output_dir: Path to output directory for memmap files
        patient_name: Name of patient (for logging)
        verbose: Print progress
        
    Returns:
        Metadata dict with shapes, dtypes, etc.
    """
    npz_path = Path(npz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Converting {patient_name}: {npz_path.name}")
        print(f"{'='*80}")
    
    # Load NPZ (will take time for large files)
    if verbose:
        print(f"Loading NPZ file...")
    
    # Try loading without pickle first (safest option)
    try:
        data = np.load(npz_path, allow_pickle=False)
    except ValueError as e:
        if "Object arrays cannot be loaded" in str(e):
            print(f"\n❌ ERROR: NPZ contains object arrays (insecure)")
            print(f"Regenerate the NPZ file using build_chbmit_windows.py")
            print(f"This ensures safe, pickle-free loading for panel submissions.")
            raise
        else:
            raise
    
    # Get arrays
    X = data['X']  # (n_windows, n_channels, n_samples)
    y = data['y']  # (n_windows,)
    
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    
    # Extract metadata if available
    meta = {}
    if 'fs' in data:
        meta['fs'] = float(data['fs'])
    if 'channel_names' in data:
        meta['channel_names'] = list(data['channel_names'])
    
    # Compute window parameters from shape
    n_windows, n_channels, n_samples = X.shape
    meta['n_windows'] = int(n_windows)
    meta['n_channels'] = int(n_channels)
    meta['n_samples_per_window'] = int(n_samples)
    meta['patient'] = patient_name
    
    # Create X memmap (float32)
    if verbose:
        print(f"Creating X memmap ({X.shape}, float32)...")
    X_memmap_path = output_dir / 'X.dat'
    X_memmap = np.memmap(
        X_memmap_path,
        dtype=np.float32,
        mode='w+',
        shape=X.shape
    )
    
    # Copy data in chunks to avoid memory spike
    chunk_size = 1000
    for i in range(0, len(X), chunk_size):
        end_i = min(i + chunk_size, len(X))
        if verbose and (i % (chunk_size * 10) == 0):
            print(f"  Progress: {i}/{len(X)}...")
        X_memmap[i:end_i] = X[i:end_i].astype(np.float32)
    
    X_memmap.flush()
    if verbose:
        print(f"  ✓ X memmap saved: {X_memmap_path} ({X_memmap_path.stat().st_size / 1e9:.2f} GB)")
    
    # Save y as npy (int8 to save space)
    if verbose:
        print(f"Creating y array (int8)...")
    y_np = (y > 0.5).astype(np.int8)  # Convert to binary
    y_path = output_dir / 'y.npy'
    np.save(y_path, y_np)
    
    n_positive = (y_np > 0).sum()
    n_negative = len(y_np) - n_positive
    meta['n_positive'] = int(n_positive)
    meta['n_negative'] = int(n_negative)
    meta['seizure_ratio_pct'] = 100.0 * n_positive / len(y_np)
    
    if verbose:
        print(f"  ✓ y saved: {y_path} (shape {y_np.shape}, dtype int8)")
        print(f"  Seizures: {n_positive} ({meta['seizure_ratio_pct']:.2f}%)")
        print(f"  Non-seizures: {n_negative}")
    
    # Save metadata
    meta_path = output_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"  ✓ Metadata saved: {meta_path}")
    
    return meta


def main():
    """Main conversion pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert NPZ files to memory-mapped arrays for realistic evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with NPZ files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for memmap files"
    )
    parser.add_argument(
        "--patients",
        nargs='+',
        default=['chb01', 'chb02'],
        help="Patients to convert"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print("\n" + "="*80)
    print("NPZ TO MEMMAP CONVERSION")
    print("="*80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Patients: {args.patients}")
    
    results = {}
    for patient in args.patients:
        npz_path = input_dir / f"{patient}.npz"
        patient_output = output_dir / patient
        
        if not npz_path.exists():
            print(f"\n⚠️  Skipping {patient}: {npz_path} not found")
            continue
        
        try:
            meta = convert_npz_to_memmap(npz_path, patient_output, patient)
            results[patient] = meta
        except Exception as e:
            print(f"\n❌ Error converting {patient}: {e}")
            return 1
    
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    for patient, meta in results.items():
        print(f"\n{patient}:")
        print(f"  Windows: {meta['n_windows']:,}")
        print(f"  Seizures: {meta['n_positive']:,} ({meta['seizure_ratio_pct']:.2f}%)")
        print(f"  Shape: {meta['n_channels']} channels × {meta['n_samples_per_window']} samples")
        print(f"  fs: {meta.get('fs', 'unknown')} Hz")
    
    print("\n✓ Conversion complete!")
    print(f"\nTo use in evaluation:")
    print(f"  python ml/training/evaluate_chbmit_realistic.py --mode realistic --train_patient chb01 --test_patient chb02")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
