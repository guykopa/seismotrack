"""Bandpass filter denoising step."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from seismotrack.interfaces.i_denoising_step import IDenoisingStep


class BandpassStep(IDenoisingStep):
    """Apply a Butterworth bandpass filter to each axis of the signal.

    Retains only the frequencies between f_low and f_high, suppressing
    both low-frequency drift and high-frequency noise. Uses a zero-phase
    implementation (sosfiltfilt) to avoid phase distortion.

    Args:
        f_low: Lower cutoff frequency in Hz.
        f_high: Upper cutoff frequency in Hz.
        sample_rate: Sampling frequency of the signal in Hz.
        order: Order of the Butterworth filter. Defaults to 4.
    """

    def __init__(
        self,
        f_low: float,
        f_high: float,
        sample_rate: float,
        order: int = 4,
    ) -> None:
        self._sos = butter(
            order,
            [f_low, f_high],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Filter each axis of the signal through the bandpass filter.

        Args:
            signal: Input array of shape (3, N). Not mutated.

        Returns:
            Filtered array of shape (3, N), dtype float64.
        """
        result = np.empty_like(signal, dtype=np.float64)
        for i in range(signal.shape[0]):
            result[i] = sosfiltfilt(self._sos, signal[i])
        return result
