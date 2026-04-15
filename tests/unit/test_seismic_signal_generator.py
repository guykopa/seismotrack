"""Unit tests for SeismicSignalGenerator — RED phase.

All tests are written before the implementation exists.
"""

import numpy as np
import pytest

from seismotrack.interfaces.i_seismic_generator import ISeismicGenerator
from seismotrack.ingestion.seismic_signal_generator import SeismicSignalGenerator


class TestSeismicSignalGeneratorShape:
    """Tests covering output shapes and dtypes."""

    def test_signal_shape_is_3_by_n(self, n_samples: int, sample_rate: float) -> None:
        """Signal must have shape (3, N) with N == n_samples."""
        gen = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate)
        signal, _ = gen.generate()
        assert signal.shape == (3, n_samples)

    def test_timestamps_shape_is_n(self, n_samples: int, sample_rate: float) -> None:
        """Timestamps must be a 1-D array of length N."""
        gen = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate)
        _, timestamps = gen.generate()
        assert timestamps.shape == (n_samples,)

    def test_signal_dtype_is_float64(self, n_samples: int, sample_rate: float) -> None:
        """Signal values must be float64."""
        gen = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate)
        signal, _ = gen.generate()
        assert signal.dtype == np.float64

    def test_timestamps_dtype_is_float64(self, n_samples: int, sample_rate: float) -> None:
        """Timestamps must be float64 Unix seconds."""
        gen = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate)
        _, timestamps = gen.generate()
        assert timestamps.dtype == np.float64


class TestSeismicSignalGeneratorSampleRate:
    """Tests covering the relationship between timestamps and sample_rate."""

    def test_timestamp_interval_matches_sample_rate(
        self, n_samples: int, sample_rate: float
    ) -> None:
        """Interval between consecutive timestamps must equal 1/sample_rate."""
        gen = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate)
        _, timestamps = gen.generate()
        expected_dt = 1.0 / sample_rate
        diffs = np.diff(timestamps)
        np.testing.assert_allclose(diffs, expected_dt, rtol=1e-9)

    def test_timestamps_start_at_zero(self, n_samples: int, sample_rate: float) -> None:
        """First timestamp must be 0.0."""
        gen = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate)
        _, timestamps = gen.generate()
        assert timestamps[0] == 0.0


class TestSeismicSignalGeneratorReproducibility:
    """Tests covering determinism via seed."""

    def test_same_seed_produces_identical_signal(
        self, n_samples: int, sample_rate: float
    ) -> None:
        """Two generators with the same seed must return identical signals."""
        gen_a = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate, seed=42)
        gen_b = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate, seed=42)
        signal_a, _ = gen_a.generate()
        signal_b, _ = gen_b.generate()
        np.testing.assert_array_equal(signal_a, signal_b)

    def test_different_seeds_produce_different_signals(
        self, n_samples: int, sample_rate: float
    ) -> None:
        """Two generators with different seeds must not produce identical signals."""
        gen_a = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate, seed=0)
        gen_b = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate, seed=1)
        signal_a, _ = gen_a.generate()
        signal_b, _ = gen_b.generate()
        assert not np.array_equal(signal_a, signal_b)


class TestSeismicSignalGeneratorInterface:
    """Tests covering Liskov / interface contract."""

    def test_implements_i_seismic_generator(
        self, n_samples: int, sample_rate: float
    ) -> None:
        """SeismicSignalGenerator must be a subtype of ISeismicGenerator."""
        gen = SeismicSignalGenerator(n_samples=n_samples, sample_rate=sample_rate)
        assert isinstance(gen, ISeismicGenerator)
