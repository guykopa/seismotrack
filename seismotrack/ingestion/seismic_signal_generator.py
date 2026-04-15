"""Synthetic seismic signal generator."""

from typing import Tuple

import numpy as np

from seismotrack.interfaces.i_seismic_generator import ISeismicGenerator


class SeismicSignalGenerator(ISeismicGenerator):
    """Generate a reproducible synthetic 3-axis seismic signal.

    Produces white Gaussian noise on each of the Z, N, E axes.
    The output is deterministic when the same seed is used.

    Args:
        n_samples: Number of samples to generate per axis.
        sample_rate: Sampling frequency in Hz.
        seed: Random seed for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        n_samples: int,
        sample_rate: float,
        seed: int = 42,
    ) -> None:
        self._n_samples = n_samples
        self._sample_rate = sample_rate
        self._seed = seed

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate signal and timestamps.

        Returns:
            A tuple (signal, timestamps) where:
                - signal has shape (3, N), dtype float64.
                - timestamps has shape (N,), dtype float64,
                  starting at 0.0 with step 1/sample_rate.
        """
        rng = np.random.default_rng(self._seed)
        signal = rng.standard_normal((3, self._n_samples)).astype(np.float64)
        timestamps = (
            np.arange(self._n_samples, dtype=np.float64) / self._sample_rate
        )
        return signal, timestamps
