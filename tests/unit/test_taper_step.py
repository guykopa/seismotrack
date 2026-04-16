"""Unit tests for TaperStep — RED phase.

All tests are written before the implementation exists.
"""

import numpy as np

from seismotrack.interfaces.i_denoising_step import IDenoisingStep
from seismotrack.denoising.taper_step import TaperStep


class TestTaperStepEdges:
    """Tests verifying that the edges of the signal are attenuated."""

    def test_first_sample_is_zero(self, raw_signal: np.ndarray) -> None:
        """First sample of each axis must be 0.0 after tapering."""
        step = TaperStep(alpha=0.1)
        result = step.apply(raw_signal)
        np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-10)

    def test_last_sample_is_zero(self, raw_signal: np.ndarray) -> None:
        """Last sample of each axis must be 0.0 after tapering."""
        step = TaperStep(alpha=0.1)
        result = step.apply(raw_signal)
        np.testing.assert_allclose(result[:, -1], 0.0, atol=1e-10)

    def test_middle_samples_unchanged(self, raw_signal: np.ndarray) -> None:
        """Samples in the flat center of the window must be unchanged."""
        step = TaperStep(alpha=0.1)
        result = step.apply(raw_signal)
        n = raw_signal.shape[1]
        taper_len = int(0.1 / 2 * n)
        center = slice(taper_len + 1, n - taper_len - 1)
        np.testing.assert_allclose(result[:, center], raw_signal[:, center], atol=1e-10)


class TestTaperStepShape:
    """Tests verifying output shape and immutability."""

    def test_output_shape_unchanged(self, raw_signal: np.ndarray) -> None:
        """Output shape must be identical to input shape."""
        step = TaperStep(alpha=0.1)
        result = step.apply(raw_signal)
        assert result.shape == raw_signal.shape

    def test_input_not_mutated(self, raw_signal: np.ndarray) -> None:
        """TaperStep must not modify the input array in place."""
        original = raw_signal.copy()
        step = TaperStep(alpha=0.1)
        step.apply(raw_signal)
        np.testing.assert_array_equal(raw_signal, original)

    def test_output_dtype_is_float64(self, raw_signal: np.ndarray) -> None:
        """Output dtype must remain float64."""
        step = TaperStep(alpha=0.1)
        result = step.apply(raw_signal)
        assert result.dtype == np.float64


class TestTaperStepAlpha:
    """Tests verifying the effect of the alpha parameter."""

    def test_alpha_zero_is_identity(self, raw_signal: np.ndarray) -> None:
        """alpha=0 produces a rectangular window — signal must be unchanged."""
        step = TaperStep(alpha=0.0)
        result = step.apply(raw_signal)
        np.testing.assert_allclose(result, raw_signal, atol=1e-10)

    def test_larger_alpha_attenuates_more_samples(
        self, raw_signal: np.ndarray
    ) -> None:
        """A larger alpha must taper more samples near the edges."""
        step_small = TaperStep(alpha=0.1)
        step_large = TaperStep(alpha=0.5)
        result_small = step_small.apply(raw_signal)
        result_large = step_large.apply(raw_signal)
        n = raw_signal.shape[1]
        quarter = n // 4
        energy_small = np.sum(result_small[:, :quarter] ** 2)
        energy_large = np.sum(result_large[:, :quarter] ** 2)
        assert energy_large < energy_small


class TestTaperStepInterface:
    """Tests covering the IDenoisingStep contract."""

    def test_implements_i_denoising_step(self) -> None:
        """TaperStep must be a subtype of IDenoisingStep."""
        step = TaperStep(alpha=0.1)
        assert isinstance(step, IDenoisingStep)
