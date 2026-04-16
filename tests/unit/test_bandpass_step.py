"""Unit tests for BandpassStep — RED phase.

All tests are written before the implementation exists.
"""

import numpy as np

from seismotrack.interfaces.i_denoising_step import IDenoisingStep
from seismotrack.denoising.bandpass_step import BandpassStep


class TestBandpassStepFrequency:
    """Tests verifying that frequencies outside the band are attenuated."""

    def test_inband_frequency_preserved(
        self, n_samples: int, sample_rate: float
    ) -> None:
        """A pure sine wave inside the passband must pass with high energy."""
        t = np.arange(n_samples) / sample_rate
        freq_in = 5.0  # Hz — inside [1, 10]
        sine = np.sin(2 * np.pi * freq_in * t).astype(np.float64)
        signal = np.stack([sine, sine, sine])
        step = BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate)
        result = step.apply(signal)
        energy_in = np.sum(result ** 2)
        energy_original = np.sum(signal ** 2)
        assert energy_in > 0.5 * energy_original

    def test_outband_low_frequency_attenuated(
        self, n_samples: int, sample_rate: float
    ) -> None:
        """A pure sine wave below f_low must be strongly attenuated."""
        t = np.arange(n_samples) / sample_rate
        freq_out = 0.1  # Hz — below f_low=1.0
        sine = np.sin(2 * np.pi * freq_out * t).astype(np.float64)
        signal = np.stack([sine, sine, sine])
        step = BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate)
        result = step.apply(signal)
        energy_out = np.sum(result ** 2)
        energy_original = np.sum(signal ** 2)
        assert energy_out < 0.1 * energy_original

    def test_outband_high_frequency_attenuated(
        self, n_samples: int, sample_rate: float
    ) -> None:
        """A pure sine wave above f_high must be strongly attenuated."""
        t = np.arange(n_samples) / sample_rate
        freq_out = 40.0  # Hz — above f_high=10.0
        sine = np.sin(2 * np.pi * freq_out * t).astype(np.float64)
        signal = np.stack([sine, sine, sine])
        step = BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate)
        result = step.apply(signal)
        energy_out = np.sum(result ** 2)
        energy_original = np.sum(signal ** 2)
        assert energy_out < 0.1 * energy_original


class TestBandpassStepShape:
    """Tests verifying output shape and immutability."""

    def test_output_shape_unchanged(self, raw_signal: np.ndarray, sample_rate: float) -> None:
        """Output shape must be identical to input shape."""
        step = BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate)
        result = step.apply(raw_signal)
        assert result.shape == raw_signal.shape

    def test_input_not_mutated(self, raw_signal: np.ndarray, sample_rate: float) -> None:
        """BandpassStep must not modify the input array in place."""
        original = raw_signal.copy()
        step = BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate)
        step.apply(raw_signal)
        np.testing.assert_array_equal(raw_signal, original)

    def test_output_dtype_is_float64(self, raw_signal: np.ndarray, sample_rate: float) -> None:
        """Output dtype must remain float64."""
        step = BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate)
        result = step.apply(raw_signal)
        assert result.dtype == np.float64


class TestBandpassStepAxes:
    """Tests verifying that all 3 axes are filtered independently."""

    def test_all_axes_are_filtered(
        self, n_samples: int, sample_rate: float
    ) -> None:
        """Each axis must be filtered — no axis left unmodified."""
        t = np.arange(n_samples) / sample_rate
        freq_out = 40.0
        sine = np.sin(2 * np.pi * freq_out * t).astype(np.float64)
        signal = np.stack([sine, sine, sine])
        step = BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate)
        result = step.apply(signal)
        for axis in range(3):
            energy = np.sum(result[axis] ** 2)
            energy_original = np.sum(signal[axis] ** 2)
            assert energy < 0.1 * energy_original


class TestBandpassStepInterface:
    """Tests covering the IDenoisingStep contract."""

    def test_implements_i_denoising_step(self, sample_rate: float) -> None:
        """BandpassStep must be a subtype of IDenoisingStep."""
        step = BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate)
        assert isinstance(step, IDenoisingStep)
