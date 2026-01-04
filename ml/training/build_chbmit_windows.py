"""
CHB-MIT Dataset Preprocessing and Windowing for Seizure Detection

This script:
1. Loads EDF files from CHB-MIT dataset (chb01, chb02)
2. Extracts EEG channels and preprocesses (bandpass filter, z-score normalization)
3. Creates sliding windows with seizure labels from summary files
4. Saves windowed data as NumPy archives (.npz)

Output: D:/epimind/ml/data/chbmit_processed/{chb01,chb02}.npz
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
import argparse

warnings.filterwarnings('ignore')

# Try to import MNE (required for EDF reading)
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("[WARNING] MNE-Python not installed. Install with: pip install mne")

from scipy import signal


# ============================================================================
# Configuration (defaults - overridable by CLI)
# ============================================================================

DATASET_ROOT = Path("D:/epimind/ml/data/chbmit")
OUTPUT_DIR = Path("D:/epimind/ml/data/chbmit_processed")
PATIENTS = ["chb01", "chb02"]

# Preprocessing parameters
BANDPASS_LOW = 0.5  # Hz
BANDPASS_HIGH = 40.0  # Hz

# Windowing parameters
WINDOW_SEC = 2.0
STEP_SEC = 1.0

# EEG channel patterns (common naming conventions in CHB-MIT)
EEG_CHANNEL_PATTERNS = [
    "FP", "F", "T", "C", "P", "O",  # Standard 10-20 system prefixes
    "EEG", "Fp", "Cz", "Pz", "Oz"
]

EXCLUDE_PATTERNS = [
    "ECG", "EMG", "EOG", "Resp", "SpO2", "HR", "BP", "EKG",
    "Photic", "IBI", "Bursts", "Suppr"
]


# ============================================================================
# Helper Functions
# ============================================================================

def is_eeg_channel(channel_name: str) -> bool:
    """
    Determine if a channel is an EEG channel.
    
    Args:
        channel_name: Channel name from EDF file
        
    Returns:
        True if channel is likely EEG
    """
    # Skip excluded channels
    for exclude in EXCLUDE_PATTERNS:
        if exclude.lower() in channel_name.lower():
            return False
    
    # Check if it matches EEG patterns
    for pattern in EEG_CHANNEL_PATTERNS:
        if pattern.lower() in channel_name.lower():
            return True
    
    return False


def parse_seizure_summary(patient: str, edf_filename: str, dataset_root: Path) -> List[Tuple[float, float]]:
    """
    Parse seizure intervals from the patient summary file.
    
    Robust parsing for CHB-MIT summary format:
    File Name: chb01_03.edf
    File Start Time: ...
    File End Time: ...
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds
    
    Args:
        patient: Patient ID (e.g., 'chb01')
        edf_filename: EDF filename (e.g., 'chb01_01.edf')
        dataset_root: Root path to dataset
        
    Returns:
        List of (start_sec, end_sec) tuples for seizures in this file
    """
    seizure_intervals = []
    
    # Read summary file
    summary_path = dataset_root / patient / f"{patient}-summary.txt"
    if not summary_path.exists():
        return seizure_intervals
    
    try:
        with open(summary_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find the section for this EDF file
        for i, line in enumerate(lines):
            if f"File Name: {edf_filename}" in line:
                # Found the file section, now extract seizure info
                # Look for "Number of Seizures in File: N"
                seizure_count = 0
                
                for j in range(i, min(i + 10, len(lines))):
                    if "Number of Seizures in File:" in lines[j]:
                        try:
                            seizure_count = int(lines[j].split(":")[-1].strip())
                        except (ValueError, IndexError):
                            seizure_count = 0
                        
                        # Now read seizure times
                        k = j + 1
                        seizures_read = 0
                        
                        while k < len(lines) and seizures_read < seizure_count:
                            if "Seizure Start Time:" in lines[k]:
                                try:
                                    start_match = re.search(r'(\d+)\s+seconds?', lines[k])
                                    if start_match:
                                        start_sec = float(start_match.group(1))
                                        
                                        # Next line should be "Seizure End Time:"
                                        if k + 1 < len(lines) and "Seizure End Time:" in lines[k + 1]:
                                            end_match = re.search(r'(\d+)\s+seconds?', lines[k + 1])
                                            if end_match:
                                                end_sec = float(end_match.group(1))
                                                if start_sec < end_sec:
                                                    seizure_intervals.append((start_sec, end_sec))
                                                    seizures_read += 1
                                                    k += 2
                                                    continue
                                except (ValueError, IndexError):
                                    pass
                            
                            k += 1
                        
                        break
                
                break
    
    except Exception as e:
        print(f"[WARNING] Error parsing summary for {edf_filename}: {e}")
    
    return seizure_intervals


def load_and_preprocess_edf(edf_path: Path) -> Optional[Tuple[np.ndarray, List[str], float]]:
    """
    Load EDF file and preprocess EEG data.
    
    Args:
        edf_path: Path to EDF file
        
    Returns:
        Tuple of (data, channel_names, fs) or None if loading fails
        - data: shape (channels, samples), float32
        - channel_names: list of EEG channel names
        - fs: sampling frequency in Hz
    """
    if not MNE_AVAILABLE:
        print(f"[ERROR] MNE required to load {edf_path.name}")
        return None
    
    try:
        # Load EDF file
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=0)
        
        # Get sampling frequency
        fs = raw.info['sfreq']
        
        # Filter channels to EEG only
        eeg_channels = [ch for ch in raw.ch_names if is_eeg_channel(ch)]
        
        if not eeg_channels:
            print(f"[WARNING] No EEG channels found in {edf_path.name}")
            return None
        
        # Get EEG data
        data = raw.get_data(picks=eeg_channels)  # shape: (channels, samples)
        
        # Bandpass filter (0.5–40 Hz)
        sos = signal.butter(4, [BANDPASS_LOW, BANDPASS_HIGH], btype='band', 
                           fs=fs, output='sos')
        data = signal.sosfilt(sos, data, axis=1)  # Apply along time axis
        
        # Z-score normalize per channel
        data = (data - data.mean(axis=1, keepdims=True)) / (
            data.std(axis=1, keepdims=True) + 1e-8
        )
        
        return data.astype(np.float32), eeg_channels, fs
    
    except Exception as e:
        print(f"[ERROR] Failed to load {edf_path.name}: {e}")
        return None


def create_sliding_windows(
    data: np.ndarray,
    fs: float,
    seizure_intervals: List[Tuple[float, float]],
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from EEG data with seizure labels.
    
    Args:
        data: EEG data, shape (channels, samples)
        fs: Sampling frequency in Hz
        seizure_intervals: List of (start_sec, end_sec) for seizures
        window_sec: Window duration in seconds
        step_sec: Step duration in seconds
        
    Returns:
        Tuple of (X, y)
        - X: shape (num_windows, channels, window_samples), float32
        - y: shape (num_windows,), int64, 1 if seizure, 0 if non-seizure
    """
    channels, total_samples = data.shape
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    X = []
    y = []
    
    # Create windows
    start_idx = 0
    while start_idx + window_samples <= total_samples:
        end_idx = start_idx + window_samples
        
        # Extract window
        window = data[:, start_idx:end_idx]
        X.append(window)
        
        # Determine label
        window_start_sec = start_idx / fs
        window_end_sec = end_idx / fs
        
        # Check if window overlaps any seizure interval
        is_seizure = 0
        for seizure_start, seizure_end in seizure_intervals:
            # Check overlap
            if not (window_end_sec <= seizure_start or window_start_sec >= seizure_end):
                is_seizure = 1
                break
        
        y.append(is_seizure)
        start_idx += step_samples
    
    X = np.array(X, dtype=np.float32)  # shape: (num_windows, channels, window_samples)
    y = np.array(y, dtype=np.int64)    # shape: (num_windows,)
    
    return X, y


def process_patient(patient: str, dataset_root: Path) -> Optional[Tuple[np.ndarray, np.ndarray, float, List[str]]]:
    """
    Process all EDF files for a patient.
    
    Args:
        patient: Patient ID (e.g., 'chb01')
        dataset_root: Root path to dataset
        
    Returns:
        Tuple of (X_all, y_all, fs, channel_names) or None if processing fails
    """
    patient_dir = dataset_root / patient
    
    if not patient_dir.exists():
        print(f"[ERROR] Patient directory not found: {patient_dir}")
        return None
    
    # Find all EDF files
    edf_files = sorted(patient_dir.glob(f"{patient}_*.edf"))
    
    if not edf_files:
        print(f"[ERROR] No EDF files found for patient {patient}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Processing patient: {patient}")
    print(f"Found {len(edf_files)} EDF files")
    print(f"{'='*70}")
    
    all_X = []
    all_y = []
    all_fs = None
    all_channel_names = None
    
    total_windows = 0
    total_seizure_windows = 0
    
    for edf_file in edf_files:
        print(f"\n[{edf_file.name}]")
        
        # Load and preprocess
        result = load_and_preprocess_edf(edf_file)
        if result is None:
            continue
        
        data, channel_names, fs = result
        
        if all_fs is None:
            all_fs = fs
        elif all_fs != fs:
            print(f"[WARNING] Sampling frequency mismatch: {all_fs} vs {fs}")
        
        if all_channel_names is None:
            all_channel_names = channel_names
        
        # Parse seizure information
        seizure_intervals = parse_seizure_summary(patient, edf_file.name, dataset_root)
        
        # Create windows
        X, y = create_sliding_windows(data, fs, seizure_intervals)
        
        num_windows = len(X)
        num_seizure_windows = np.sum(y)
        
        print(f"  → Data shape: {data.shape}")
        print(f"  → Windows created: {num_windows}")
        print(f"  → Seizure windows: {num_seizure_windows}")
        print(f"  → Non-seizure windows: {num_windows - num_seizure_windows}")
        
        if seizure_intervals:
            print(f"  → Seizure intervals found: {len(seizure_intervals)}")
            for start, end in seizure_intervals:
                print(f"     {start:.1f}s - {end:.1f}s")
        else:
            print(f"  → No seizure intervals in this file")
        
        all_X.append(X)
        all_y.append(y)
        total_windows += num_windows
        total_seizure_windows += num_seizure_windows
    
    if not all_X:
        print(f"[ERROR] No valid data for patient {patient}")
        return None
    
    # Concatenate all windows
    X_all = np.concatenate(all_X, axis=0)  # shape: (total_windows, channels, timesteps)
    y_all = np.concatenate(all_y, axis=0)  # shape: (total_windows,)
    
    print(f"\n{'-'*70}")
    print(f"Patient {patient} Summary:")
    print(f"  Total windows: {total_windows}")
    print(f"  Total seizure windows: {total_seizure_windows}")
    print(f"  Total non-seizure windows: {total_windows - total_seizure_windows}")
    print(f"  Seizure ratio: {total_seizure_windows/total_windows*100:.2f}%")
    print(f"  Final X shape: {X_all.shape}")
    print(f"  Final y shape: {y_all.shape}")
    print(f"  Sampling frequency: {all_fs} Hz")
    print(f"  Channel count: {len(all_channel_names)}")
    print(f"{'-'*70}")
    
    return X_all, y_all, all_fs, all_channel_names


def main():
    """Main preprocessing pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="CHB-MIT dataset preprocessing with seizure labeling"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATASET_ROOT,
        help=f"Input dataset root directory (default: {DATASET_ROOT})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for processed data (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=PATIENTS,
        help="Patient IDs to process (default: chb01 chb02)"
    )
    
    args = parser.parse_args()
    
    # Use CLI arguments to override defaults
    input_root = args.input
    output_dir = args.output
    patients = args.patients
    
    print("\n" + "="*70)
    print("CHB-MIT Dataset Preprocessing")
    print("="*70)
    print(f"\nInput root: {input_root}")
    print(f"Output directory: {output_dir}")
    print(f"Patients: {patients}")
    
    # Check MNE availability
    if not MNE_AVAILABLE:
        print("[ERROR] MNE-Python is required. Install with: pip install mne scipy")
        return
    
    # Check dataset directory
    if not input_root.exists():
        print(f"[ERROR] Dataset root not found: {input_root}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory ready: {output_dir}")
    
    # Track overall statistics
    overall_positive = 0
    overall_negative = 0
    
    # Process each patient
    for patient in patients:
        
        result = process_patient(patient, input_root)
        
        if result is None:
            print(f"[SKIP] Could not process patient {patient}")
            continue
        
        X, y, fs, channel_names = result
        
        # Enforce numeric dtypes for safety and pickle-free loading
        X = X.astype(np.float32)
        y = y.astype(np.int8)
        
        # Track label distribution
        num_positive = np.sum(y)
        num_negative = len(y) - num_positive
        overall_positive += num_positive
        overall_negative += num_negative
        
        # Save as NPZ (numeric arrays only - no objects)
        output_path = output_dir / f"{patient}.npz"
        
        try:
            np.savez(
                output_path,
                X=X,
                y=y,
                fs=fs
            )
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n[SUCCESS] Saved {patient} to {output_path}")
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"  Label distribution: {num_positive} positive, {num_negative} negative")
        
        except Exception as e:
            print(f"\n[ERROR] Failed to save {patient}: {e}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("Preprocessing complete!")
    print(f"{'='*70}")
    print(f"\nOVERALL LABEL DISTRIBUTION:")
    print(f"  Positive (seizure) windows: {overall_positive}")
    print(f"  Negative (non-seizure) windows: {overall_negative}")
    print(f"  Total windows: {overall_positive + overall_negative}")
    if overall_positive + overall_negative > 0:
        print(f"  Seizure ratio: {overall_positive/(overall_positive + overall_negative)*100:.2f}%")
    
    # CRITICAL VALIDATION
    if overall_positive == 0:
        print(f"\n⚠️  WARNING: No positive windows found!")
        print(f"This suggests seizure label extraction may have failed.")
        print(f"Check that summary files exist and contain seizure data.")
    else:
        print(f"\n✓ Seizure labels successfully extracted!")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
