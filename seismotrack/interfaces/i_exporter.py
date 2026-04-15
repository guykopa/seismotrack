"""Abstract interface for signal exporters."""

from abc import ABC, abstractmethod

import numpy as np


class IExporter(ABC):
    """Contract for any class that persists a processed seismic signal.

    Implementations handle all I/O concerns (file format, metadata
    encoding) without knowledge of upstream processing.
    """

    @abstractmethod
    def export(
        self,
        signal: np.ndarray,
        timestamps: np.ndarray,
        output_path: str,
    ) -> None:
        """Persist the signal and timestamps to a file.

        Args:
            signal: Processed signal array of shape (3, N), units nm/s.
            timestamps: Unix timestamps array of shape (N,), dtype float64.
            output_path: Destination file path (created if absent).
        """
