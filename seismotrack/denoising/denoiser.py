"""Denoising orchestrator."""

from typing import List

import numpy as np

from seismotrack.interfaces.i_denoising_step import IDenoisingStep


class Denoiser:
    """Apply a sequence of denoising steps to a seismic signal.

    Receives its steps by injection and applies them in order.
    Never instantiates concrete step classes internally.

    Args:
        steps: Ordered list of IDenoisingStep to apply.
    """

    def __init__(self, steps: List[IDenoisingStep]) -> None:
        self._steps = steps

    def denoise(self, signal: np.ndarray) -> np.ndarray:
        """Apply all steps in order and return the processed signal.

        Args:
            signal: Input array of shape (3, N). Not mutated.

        Returns:
            Processed array of shape (3, N).
        """
        result = signal
        for step in self._steps:
            result = step.apply(result)
        return result
