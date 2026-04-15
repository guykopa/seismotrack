"""Abstract interface for a single denoising step."""

from abc import ABC, abstractmethod

import numpy as np


class IDenoisingStep(ABC):
    """Contract that every denoising step must fulfil.

    Each concrete step receives a raw signal array, applies one
    transformation, and returns the transformed signal without
    side effects on the input.
    """

    @abstractmethod
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply the denoising transformation to the signal.

        Args:
            signal: Input signal array of shape (3, N), where axis 0
                corresponds to axes Z, N, E and N is the number of samples.

        Returns:
            Transformed signal with the same shape (3, N).
        """
