"""Abstract interface for seismic signal generators."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class ISeismicGenerator(ABC):
    """Contract for any class that produces a synthetic seismic signal.

    Implementations are responsible for generating both the signal data
    and the corresponding timestamps array.
    """

    @abstractmethod
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a seismic signal and its timestamps.

        Returns:
            A tuple (signal, timestamps) where:
                - signal has shape (3, N), dtype float64, units nm/s.
                  Axis 0 = Z, axis 1 = N, axis 2 = E.
                - timestamps has shape (N,), dtype float64,
                  representing Unix seconds.
        """
