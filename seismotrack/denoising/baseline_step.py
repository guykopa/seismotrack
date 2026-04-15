"""Baseline correction denoising step."""

import numpy as np

from seismotrack.interfaces.i_denoising_step import IDenoisingStep


class BaselineStep(IDenoisingStep):
    """Remove the DC offset from each axis of the signal.

    Subtracts the per-axis mean so that each channel oscillates
    around zero. This is the first step in the denoising chain.
    """

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Subtract the mean of each axis from the signal.

        Args:
            signal: Input array of shape (3, N). Not mutated.

        Returns:
            Baseline-corrected array of shape (3, N) with zero mean
            on each axis.
        """
        return signal - signal.mean(axis=1, keepdims=True)
