"""Unit tests for BaselineStep — RED phase.

All tests are written before the implementation exists.
"""

import numpy as np

from seismotrack.interfaces.i_denoising_step import IDenoisingStep
from seismotrack.denoising.baseline_step import BaselineStep


class TestBaselineStepDCRemoval:
    """Tests verifying that the DC offset is correctly removed."""

    def test_dc_offset_removed_all_axes(self, raw_signal: np.ndarray) -> None:
        """Mean of each axis must be ~0 after applying BaselineStep."""
        step = BaselineStep()
        result = step.apply(raw_signal)
        means = result.mean(axis=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_negative_offset_removed(self) -> None:
        """BaselineStep must handle signals with negative DC offset."""
        signal = np.full((3, 100), -7.0) + np.random.default_rng(0).standard_normal((3, 100))
        step = BaselineStep()
        result = step.apply(signal)
        np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-10)

    def test_zero_offset_is_identity(self) -> None:
        """A signal already centered on zero must be unchanged."""
        rng = np.random.default_rng(99)
        signal = rng.standard_normal((3, 500))
        signal -= signal.mean(axis=1, keepdims=True)  # force zero mean
        step = BaselineStep()
        result = step.apply(signal)
        np.testing.assert_allclose(result, signal, atol=1e-12)


class TestBaselineStepShape:
    """Tests verifying that the output shape is preserved."""

    def test_output_shape_unchanged(self, raw_signal: np.ndarray) -> None:
        """Output shape must be identical to input shape."""
        step = BaselineStep()
        result = step.apply(raw_signal)
        assert result.shape == raw_signal.shape

    def test_input_not_mutated(self, raw_signal: np.ndarray) -> None:
        """BaselineStep must not modify the input array in place."""
        original = raw_signal.copy()
        step = BaselineStep()
        step.apply(raw_signal)
        np.testing.assert_array_equal(raw_signal, original)


class TestBaselineStepInterface:
    """Tests covering the IDenoisingStep contract."""

    def test_implements_i_denoising_step(self) -> None:
        """BaselineStep must be a subtype of IDenoisingStep."""
        step = BaselineStep()
        assert isinstance(step, IDenoisingStep)
