"""Taper denoising step."""

import numpy as np
from scipy.signal.windows import tukey

from seismotrack.interfaces.i_denoising_step import IDenoisingStep


class TaperStep(IDenoisingStep):
    """Apply a Tukey window to attenuate the edges of the signal.

    Multiplies each axis by a Tukey (tapered cosine) window so that
    the signal fades to zero at both ends. This prevents edge artefacts
    when the bandpass filter is applied in the next step.

    Args:
        alpha: Shape parameter of the Tukey window in [0, 1].
            0 produces a rectangular window (no tapering).
            1 produces a full Hann window.
            Defaults to 0.1.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Multiply the signal by a Tukey window along the time axis.

        Args:
            signal: Input array of shape (3, N). Not mutated.

        Returns:
            Tapered array of shape (3, N), dtype float64.
        """
        n = signal.shape[1]
        window = tukey(n, alpha=self._alpha).astype(np.float64)
        return (signal * window).astype(np.float64)
