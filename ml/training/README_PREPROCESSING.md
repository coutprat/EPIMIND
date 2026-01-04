# CHB-MIT EEG Preprocessing for Seizure Detection

## Overview

This preprocessing pipeline transforms raw CHB-MIT EDF files into windowed, labeled datasets suitable for machine learning.

### What it does:

1. **Loads EDF files** from `D:/epimind/ml/data/chbmit/chb01/` and `chb02/` using MNE-Python
2. **Filters EEG channels** (excludes ECG, EMG, EOG, etc.)
3. **Applies signal preprocessing:**
   - Bandpass filter (0.5–40 Hz)
   - Z-score normalization per channel
4. **Creates sliding windows:**
   - Window size: 2 seconds
   - Step size: 1 second
   - Shape: (num_windows, channels, timesteps)
5. **Labels windows** based on seizure intervals from summary files:
   - Label = 1 if window overlaps seizure
   - Label = 0 otherwise
6. **Saves output** as NumPy archives (.npz) with:
   - `X`: windowed EEG data (float32)
   - `y`: seizure labels (int64)
   - `fs`: sampling frequency
   - `channel_names`: EEG channel names

### Output files:

```
D:/epimind/ml/data/chbmit_processed/
├── chb01.npz          # Patient 1 processed data
└── chb02.npz          # Patient 2 processed data
```

Each `.npz` contains:
- **X** (ndarray): shape (num_windows, channels, window_samples), dtype float32
- **y** (ndarray): shape (num_windows,), dtype int64, values [0, 1]
- **fs** (float): sampling frequency in Hz
- **channel_names** (array): list of EEG channel names

---

## Installation & Setup

### 1. Install MNE and dependencies:

```bash
cd d:\epimind
.\.venv\Scripts\python.exe -m pip install mne scipy
```

### 2. Verify dataset structure:

```
D:/epimind/ml/data/chbmit/
├── chb01/
│   ├── chb01_01.edf
│   ├── chb01_02.edf
│   ├── ... (up to 40+ files)
│   └── chb01-summary.txt
└── chb02/
    ├── chb02_01.edf
    ├── chb02_02.edf
    ├── ... (up to 25+ files)
    └── chb02-summary.txt
```

---

## Running the Preprocessing

```bash
cd d:\epimind
$env:PYTHONPATH = 'd:\epimind\backend'
d:\epimind\backend\.venv\Scripts\python.exe ml/training/build_chbmit_windows.py
```

### Expected Output:

```
======================================================================
CHB-MIT Dataset Preprocessing
======================================================================

======================================================================
Processing patient: chb01
Found 34 EDF files
======================================================================

[chb01_01.edf]
  → Data shape: (23, 921600)
  → Windows created: 768
  → Seizure windows: 0
  → Non-seizure windows: 768
  → Seizure intervals found: 0

[chb01_02.edf]
  → Data shape: (23, 921600)
  → Windows created: 768
  → Seizure windows: 64
  → Non-seizure windows: 704
  → Seizure intervals found: 1
     2573.0s - 2703.0s

... (more files)

----------------------------------------------------------------------
Patient chb01 Summary:
  Total windows: 25728
  Total seizure windows: 1024
  Total non-seizure windows: 24704
  Seizure ratio: 3.98%
  Final X shape: (25728, 23, 512)
  Final y shape: (25728,)
  Sampling frequency: 256 Hz
  Channel count: 23
----------------------------------------------------------------------

[SUCCESS] Saved chb01 to D:/epimind/ml/data/chbmit_processed/chb01.npz
  File size: 2847.33 MB

... (similar for chb02)

======================================================================
Preprocessing complete!
======================================================================
```

---

## Using Processed Data for Training

### Loading the data:

```python
import numpy as np

# Load preprocessed data
data = np.load('D:/epimind/ml/data/chbmit_processed/chb01.npz', allow_pickle=True)

X = data['X']           # shape: (25728, 23, 512)
y = data['y']           # shape: (25728,)
fs = float(data['fs'])  # 256.0 Hz
channels = data['channel_names']

print(f"Samples: {len(X)}")
print(f"Channels: {X.shape[1]}")
print(f"Timesteps per window: {X.shape[2]}")
print(f"Seizure ratio: {y.mean()*100:.2f}%")
```

### Data characteristics:

- **X**: float32, normalized to zero-mean unit-variance per channel
- **y**: int64, binary classification (0=non-seizure, 1=seizure)
- **Window duration**: 2 seconds at 256 Hz = 512 timesteps
- **Channels**: Typically 19–23 EEG channels depending on file
- **Seizure imbalance**: ~4% seizure windows (realistic distribution)

---

## Customization

Edit `build_chbmit_windows.py` to change:

```python
# Bandpass filter frequency range
BANDPASS_LOW = 0.5      # Hz
BANDPASS_HIGH = 40.0    # Hz

# Window parameters
WINDOW_SEC = 2.0        # seconds
STEP_SEC = 1.0          # seconds (50% overlap)
```

---

## Troubleshooting

### Issue: "No EEG channels found"
- Some files may have non-standard channel naming.
- Edit `EEG_CHANNEL_PATTERNS` in the script to include more patterns.

### Issue: "No seizure intervals in this file"
- This is expected; not all files contain seizures.
- Check the summary file for seizure information.

### Issue: "Sampling frequency mismatch"
- CHB-MIT typically uses 256 Hz; minor mismatches are logged as warnings.
- The script uses the first file's frequency for the dataset.

### Issue: MNE ImportError
```bash
pip install mne scipy
```

---

## References

- **Dataset**: [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
- **MNE Documentation**: https://mne.tools/
- **Window-based seizure detection**: Common approach in EEG preprocessing

---

## Next Steps

After preprocessing, the data is ready for:
1. Model training (CNN, RNN, Transformer-based architectures)
2. Feature extraction (spectral, temporal)
3. Cross-validation and performance evaluation
