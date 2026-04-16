"""HDF5 signal exporter."""

import pathlib

import h5py
import numpy as np

from seismotrack.interfaces.i_exporter import IExporter


class HDF5Exporter(IExporter):
    """Persist a processed seismic signal to an HDF5 file.

    Creates two datasets:
        - ``signal``: shape (3, N), float64, units nm/s, axes Z,N,E.
        - ``timestamps``: shape (N,), float64, units s.

    Intermediate directories are created automatically if absent.
    """

    def export(
        self,
        signal: np.ndarray,
        timestamps: np.ndarray,
        output_path: str,
    ) -> None:
        """Write signal and timestamps to an HDF5 file.

        Args:
            signal: Processed signal array of shape (3, N), units nm/s.
            timestamps: Unix timestamps array of shape (N,), dtype float64.
            output_path: Destination file path (created if absent).
        """
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            ds_signal = f.create_dataset(
                "signal", data=signal.astype(np.float64)
            )
            ds_signal.attrs["units"] = "nm/s"
            ds_signal.attrs["axes"] = "Z,N,E"

            ds_ts = f.create_dataset(
                "timestamps", data=timestamps.astype(np.float64)
            )
            ds_ts.attrs["units"] = "s"
