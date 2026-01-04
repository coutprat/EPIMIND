# ml/training/datasets.py

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class SyntheticEEGDataset(Dataset):
    """
    Generates synthetic EEG-like windows.
    - Normal windows: low amplitude noise
    - Seizure-like windows: add high-energy burst in a random segment
    Label: 0.0 (normal) or 1.0 (seizure/high-risk)
    """

    def __init__(
        self,
        n_samples: int = 10000,
        n_channels: int = 4,
        window_size: int = 512,
        seizure_prob: float = 0.3,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.window_size = window_size
        self.seizure_prob = seizure_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def _generate_window(self, seizure: bool) -> np.ndarray:
        # Base noise
        x = self.rng.normal(0.0, 1.0, size=(self.n_channels, self.window_size))

        if seizure:
            # Add a stronger burst in a random subsegment
            burst_len = self.window_size // 8
            start = self.rng.integers(0, self.window_size - burst_len)
            end = start + burst_len

            amp = self.rng.uniform(3.0, 6.0)
            burst = self.rng.normal(amp, 0.5, size=(self.n_channels, burst_len))
            x[:, start:end] += burst

        # Normalize per-channel (zero mean, unit std) like inference preprocess
        x = x.astype("float32")
        for c in range(self.n_channels):
            std = np.std(x[c])
            if std > 1e-6:
                x[c] = (x[c] - np.mean(x[c])) / std

        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seizure = self.rng.random() < self.seizure_prob
        x = self._generate_window(seizure)
        y = np.array([1.0 if seizure else 0.0], dtype="float32")

        return torch.from_numpy(x), torch.from_numpy(y)
