"""Audio augmentation by injecting Gaussian noise."""

from __future__ import annotations

import numpy as np


def inject_noise(waveform: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    signal_power = np.mean(waveform**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(scale=np.sqrt(noise_power), size=waveform.shape)
    return waveform + noise


__all__ = ["inject_noise"]
