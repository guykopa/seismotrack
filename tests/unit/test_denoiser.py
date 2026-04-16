"""Unit tests for Denoiser — RED phase.

All tests are written before the implementation exists.
"""

import numpy as np

from seismotrack.interfaces.i_denoising_step import IDenoisingStep
from seismotrack.denoising.denoiser import Denoiser


class _DoubleStep(IDenoisingStep):
    """Test stub: multiplies the signal by 2."""

    def apply(self, signal: np.ndarray) -> np.ndarray:
        return signal * 2.0


class _AddOneStep(IDenoisingStep):
    """Test stub: adds 1 to every sample."""

    def apply(self, signal: np.ndarray) -> np.ndarray:
        return signal + 1.0


class _RecordCallStep(IDenoisingStep):
    """Test stub: records call order via a shared list."""

    def __init__(self, label: str, log: list) -> None:
        self._label = label
        self._log = log

    def apply(self, signal: np.ndarray) -> np.ndarray:
        self._log.append(self._label)
        return signal


class TestDenoiserOrdering:
    """Tests verifying that steps are applied in the correct order."""

    def test_steps_applied_in_order(self, raw_signal: np.ndarray) -> None:
        """Steps must be applied in the order they were passed."""
        log: list = []
        steps = [
            _RecordCallStep("first", log),
            _RecordCallStep("second", log),
            _RecordCallStep("third", log),
        ]
        denoiser = Denoiser(steps=steps)
        denoiser.denoise(raw_signal)
        assert log == ["first", "second", "third"]

    def test_two_steps_compose_correctly(self, raw_signal: np.ndarray) -> None:
        """double then add_one must equal signal*2 + 1."""
        denoiser = Denoiser(steps=[_DoubleStep(), _AddOneStep()])
        result = denoiser.denoise(raw_signal)
        expected = raw_signal * 2.0 + 1.0
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_order_matters(self, raw_signal: np.ndarray) -> None:
        """add_one then double must differ from double then add_one."""
        result_a = Denoiser(steps=[_DoubleStep(), _AddOneStep()]).denoise(raw_signal)
        result_b = Denoiser(steps=[_AddOneStep(), _DoubleStep()]).denoise(raw_signal)
        assert not np.allclose(result_a, result_b)


class TestDenoiserEdgeCases:
    """Tests covering empty step list and single step."""

    def test_empty_steps_returns_signal_unchanged(
        self, raw_signal: np.ndarray
    ) -> None:
        """No steps must return a signal equal to the input."""
        denoiser = Denoiser(steps=[])
        result = denoiser.denoise(raw_signal)
        np.testing.assert_array_equal(result, raw_signal)

    def test_single_step_applied(self, raw_signal: np.ndarray) -> None:
        """A single step must be applied exactly once."""
        denoiser = Denoiser(steps=[_DoubleStep()])
        result = denoiser.denoise(raw_signal)
        np.testing.assert_allclose(result, raw_signal * 2.0, atol=1e-12)


class TestDenoiserImmutability:
    """Tests verifying the input signal is never mutated."""

    def test_input_not_mutated(self, raw_signal: np.ndarray) -> None:
        """Denoiser must not modify the input array in place."""
        original = raw_signal.copy()
        denoiser = Denoiser(steps=[_DoubleStep(), _AddOneStep()])
        denoiser.denoise(raw_signal)
        np.testing.assert_array_equal(raw_signal, original)


class TestDenoiserShape:
    """Tests verifying output shape and dtype."""

    def test_output_shape_unchanged(self, raw_signal: np.ndarray) -> None:
        """Output shape must be identical to input shape."""
        denoiser = Denoiser(steps=[_DoubleStep()])
        result = denoiser.denoise(raw_signal)
        assert result.shape == raw_signal.shape
