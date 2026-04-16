"""Integration tests for Pipeline — RED phase.

All tests are written before the implementation exists.
Tests exercise the full generator → denoiser → exporter chain.
"""

import numpy as np
import h5py

from seismotrack.interfaces.i_seismic_generator import ISeismicGenerator
from seismotrack.interfaces.i_exporter import IExporter
from seismotrack.denoising.denoiser import Denoiser
from seismotrack.denoising.baseline_step import BaselineStep
from seismotrack.denoising.taper_step import TaperStep
from seismotrack.denoising.bandpass_step import BandpassStep
from seismotrack.ingestion.seismic_signal_generator import SeismicSignalGenerator
from seismotrack.export.hdf5_exporter import HDF5Exporter
from seismotrack.pipeline.pipeline import Pipeline


class _FixedGenerator(ISeismicGenerator):
    """Test stub: returns a fixed signal and timestamps."""

    def __init__(self, signal: np.ndarray, timestamps: np.ndarray) -> None:
        self._signal = signal
        self._timestamps = timestamps

    def generate(self):
        return self._signal, self._timestamps


class _CapturingExporter(IExporter):
    """Test stub: captures what was exported instead of writing to disk."""

    def __init__(self) -> None:
        self.signal: np.ndarray | None = None
        self.timestamps: np.ndarray | None = None
        self.output_path: str | None = None

    def export(
        self,
        signal: np.ndarray,
        timestamps: np.ndarray,
        output_path: str,
    ) -> None:
        self.signal = signal
        self.timestamps = timestamps
        self.output_path = output_path


class TestPipelineRun:
    """Tests verifying that run() wires all components correctly."""

    def test_run_calls_exporter(
        self, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """run() must invoke the exporter with a signal and timestamps."""
        exporter = _CapturingExporter()
        pipeline = Pipeline(
            generator=_FixedGenerator(raw_signal, timestamps),
            denoiser=Denoiser(steps=[]),
            exporter=exporter,
            output_path="out.h5",
        )
        pipeline.run()
        assert exporter.signal is not None
        assert exporter.timestamps is not None

    def test_run_passes_correct_output_path(
        self, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """run() must forward the configured output_path to the exporter."""
        exporter = _CapturingExporter()
        pipeline = Pipeline(
            generator=_FixedGenerator(raw_signal, timestamps),
            denoiser=Denoiser(steps=[]),
            exporter=exporter,
            output_path="my/output.h5",
        )
        pipeline.run()
        assert exporter.output_path == "my/output.h5"

    def test_run_passes_timestamps_unchanged(
        self, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """run() must forward timestamps from the generator to the exporter."""
        exporter = _CapturingExporter()
        pipeline = Pipeline(
            generator=_FixedGenerator(raw_signal, timestamps),
            denoiser=Denoiser(steps=[]),
            exporter=exporter,
            output_path="out.h5",
        )
        pipeline.run()
        np.testing.assert_array_equal(exporter.timestamps, timestamps)

    def test_run_applies_denoising(
        self, raw_signal: np.ndarray, timestamps: np.ndarray
    ) -> None:
        """The signal passed to the exporter must have been denoised."""
        exporter = _CapturingExporter()
        pipeline = Pipeline(
            generator=_FixedGenerator(raw_signal, timestamps),
            denoiser=Denoiser(steps=[BaselineStep()]),
            exporter=exporter,
            output_path="out.h5",
        )
        pipeline.run()
        expected = raw_signal - raw_signal.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(exporter.signal, expected, atol=1e-12)


class TestPipelineEndToEnd:
    """Full pipeline test: generator → denoiser → HDF5 file."""

    def test_full_pipeline_creates_valid_hdf5(
        self, tmp_path, n_samples: int, sample_rate: float
    ) -> None:
        """A full pipeline run must produce a readable HDF5 file."""
        output = str(tmp_path / "signal.h5")
        pipeline = Pipeline(
            generator=SeismicSignalGenerator(
                n_samples=n_samples, sample_rate=sample_rate, seed=0
            ),
            denoiser=Denoiser(steps=[
                BaselineStep(),
                TaperStep(alpha=0.1),
                BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate),
            ]),
            exporter=HDF5Exporter(),
            output_path=output,
        )
        pipeline.run()

        with h5py.File(output, "r") as f:
            assert "signal" in f
            assert "timestamps" in f
            assert f["signal"].shape == (3, n_samples)
            assert f["timestamps"].shape == (n_samples,)

    def test_full_pipeline_signal_is_denoised(
        self, tmp_path, n_samples: int, sample_rate: float
    ) -> None:
        """After the full pipeline, the signal mean must be near zero."""
        output = str(tmp_path / "signal.h5")
        pipeline = Pipeline(
            generator=SeismicSignalGenerator(
                n_samples=n_samples, sample_rate=sample_rate, seed=0
            ),
            denoiser=Denoiser(steps=[
                BaselineStep(),
                TaperStep(alpha=0.1),
                BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate),
            ]),
            exporter=HDF5Exporter(),
            output_path=output,
        )
        pipeline.run()

        with h5py.File(output, "r") as f:
            signal = f["signal"][:]
        np.testing.assert_allclose(signal.mean(axis=1), 0.0, atol=1e-2)
