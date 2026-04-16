"""Integration tests for HDF5Exporter — RED phase.

All tests are written before the implementation exists.
Tests exercise real file I/O using pytest's tmp_path fixture.
"""

import numpy as np
import h5py

from seismotrack.interfaces.i_exporter import IExporter
from seismotrack.export.hdf5_exporter import HDF5Exporter


class TestHDF5ExporterFileCreation:
    """Tests verifying that the output file is created correctly."""

    def test_file_is_created(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """Export must create a file at the given output path."""
        output = str(tmp_path / "output.h5")
        exporter = HDF5Exporter()
        exporter.export(raw_signal, timestamps, output)
        assert (tmp_path / "output.h5").exists()

    def test_creates_parent_directory(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """Export must create intermediate directories if they do not exist."""
        output = str(tmp_path / "subdir" / "output.h5")
        exporter = HDF5Exporter()
        exporter.export(raw_signal, timestamps, output)
        assert (tmp_path / "subdir" / "output.h5").exists()


class TestHDF5ExporterDatasets:
    """Tests verifying stored signal and timestamps datasets."""

    def test_signal_dataset_shape(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The 'signal' dataset must have shape (3, N)."""
        output = str(tmp_path / "output.h5")
        HDF5Exporter().export(raw_signal, timestamps, output)
        with h5py.File(output, "r") as f:
            assert f["signal"].shape == raw_signal.shape

    def test_signal_dataset_values(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The 'signal' dataset must contain the exact input values."""
        output = str(tmp_path / "output.h5")
        HDF5Exporter().export(raw_signal, timestamps, output)
        with h5py.File(output, "r") as f:
            np.testing.assert_array_equal(f["signal"][:], raw_signal)

    def test_timestamps_dataset_shape(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The 'timestamps' dataset must have shape (N,)."""
        output = str(tmp_path / "output.h5")
        HDF5Exporter().export(raw_signal, timestamps, output)
        with h5py.File(output, "r") as f:
            assert f["timestamps"].shape == timestamps.shape

    def test_timestamps_dataset_values(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The 'timestamps' dataset must contain the exact input values."""
        output = str(tmp_path / "output.h5")
        HDF5Exporter().export(raw_signal, timestamps, output)
        with h5py.File(output, "r") as f:
            np.testing.assert_array_equal(f["timestamps"][:], timestamps)

    def test_signal_dtype_is_float64(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The 'signal' dataset must be stored as float64."""
        output = str(tmp_path / "output.h5")
        HDF5Exporter().export(raw_signal, timestamps, output)
        with h5py.File(output, "r") as f:
            assert f["signal"].dtype == np.float64

    def test_timestamps_dtype_is_float64(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The 'timestamps' dataset must be stored as float64."""
        output = str(tmp_path / "output.h5")
        HDF5Exporter().export(raw_signal, timestamps, output)
        with h5py.File(output, "r") as f:
            assert f["timestamps"].dtype == np.float64


class TestHDF5ExporterMetadata:
    """Tests verifying scientific metadata attributes."""

    def test_signal_units_attribute(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The 'signal' dataset must carry a 'units' attribute equal to 'nm/s'."""
        output = str(tmp_path / "output.h5")
        HDF5Exporter().export(raw_signal, timestamps, output)
        with h5py.File(output, "r") as f:
            assert f["signal"].attrs["units"] == "nm/s"

    def test_timestamps_units_attribute(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The 'timestamps' dataset must carry a 'units' attribute equal to 's'."""
        output = str(tmp_path / "output.h5")
        HDF5Exporter().export(raw_signal, timestamps, output)
        with h5py.File(output, "r") as f:
            assert f["timestamps"].attrs["units"] == "s"

    def test_axes_attribute(
        self, tmp_path, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The 'signal' dataset must carry an 'axes' attribute equal to 'Z,N,E'."""
        output = str(tmp_path / "output.h5")
        HDF5Exporter().export(raw_signal, timestamps, output)
        with h5py.File(output, "r") as f:
            assert f["signal"].attrs["axes"] == "Z,N,E"


class TestHDF5ExporterInterface:
    """Tests covering the IExporter contract."""

    def test_implements_i_exporter(self) -> None:
        """HDF5Exporter must be a subtype of IExporter."""
        exporter = HDF5Exporter()
        assert isinstance(exporter, IExporter)
