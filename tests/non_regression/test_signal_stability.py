"""Non-regression tests for signal stability.

These tests freeze the scientific metrics of the pipeline output.
They fail when a code change silently alters the physical quality of
the processed signal, even if all unit tests still pass.

Reference values were computed with seed=42, n_samples=1000,
sample_rate=100.0 Hz, f_low=1.0 Hz, f_high=10.0 Hz.
"""

import numpy as np
import pytest
from scipy.signal import welch

from seismotrack.ingestion.seismic_signal_generator import SeismicSignalGenerator
from seismotrack.denoising.denoiser import Denoiser
from seismotrack.denoising.baseline_step import BaselineStep
from seismotrack.denoising.taper_step import TaperStep
from seismotrack.denoising.bandpass_step import BandpassStep


@pytest.fixture(scope="module")
def denoised_signal():
    """Run the full denoising chain once for all stability tests."""
    n_samples = 1000
    sample_rate = 100.0
    generator = SeismicSignalGenerator(
        n_samples=n_samples, sample_rate=sample_rate, seed=42
    )
    signal, _ = generator.generate()
    denoiser = Denoiser(steps=[
        BaselineStep(),
        TaperStep(alpha=0.1),
        BandpassStep(f_low=1.0, f_high=10.0, sample_rate=sample_rate),
    ])
    return denoiser.denoise(signal)


class TestRMSStability:
    """Tests verifying per-axis RMS amplitude stays within reference bounds."""

    def test_rms_z_axis_stable(self, denoised_signal: np.ndarray) -> None:
        """RMS of Z axis must remain close to reference value 0.4208."""
        rms = np.sqrt(np.mean(denoised_signal[0] ** 2))
        assert abs(rms - 0.42075887) < 0.05

    def test_rms_n_axis_stable(self, denoised_signal: np.ndarray) -> None:
        """RMS of N axis must remain close to reference value 0.4115."""
        rms = np.sqrt(np.mean(denoised_signal[1] ** 2))
        assert abs(rms - 0.41148496) < 0.05

    def test_rms_e_axis_stable(self, denoised_signal: np.ndarray) -> None:
        """RMS of E axis must remain close to reference value 0.4120."""
        rms = np.sqrt(np.mean(denoised_signal[2] ** 2))
        assert abs(rms - 0.41201185) < 0.05


class TestEnergyStability:
    """Tests verifying total signal energy stays within reference bounds."""

    def test_total_energy_stable(self, denoised_signal: np.ndarray) -> None:
        """Total signal energy must remain close to reference value 516.1."""
        energy = np.sum(denoised_signal ** 2)
        assert abs(energy - 516.1116604) < 50.0


class TestInBandEnergyRatio:
    """Tests verifying that most energy stays within the passband [1, 10] Hz."""

    def _inband_ratio(self, axis: np.ndarray, sample_rate: float = 100.0) -> float:
        """Compute the fraction of PSD energy within [1, 10] Hz."""
        freqs, psd = welch(axis, fs=sample_rate, nperseg=256)
        inband = np.sum(psd[(freqs >= 1.0) & (freqs <= 10.0)])
        return float(inband / np.sum(psd))

    def test_inband_ratio_z_above_threshold(
        self, denoised_signal: np.ndarray
    ) -> None:
        """At least 90% of Z axis energy must be within [1, 10] Hz."""
        assert self._inband_ratio(denoised_signal[0]) > 0.90

    def test_inband_ratio_n_above_threshold(
        self, denoised_signal: np.ndarray
    ) -> None:
        """At least 90% of N axis energy must be within [1, 10] Hz."""
        assert self._inband_ratio(denoised_signal[1]) > 0.90

    def test_inband_ratio_e_above_threshold(
        self, denoised_signal: np.ndarray
    ) -> None:
        """At least 90% of E axis energy must be within [1, 10] Hz."""
        assert self._inband_ratio(denoised_signal[2]) > 0.90


class TestDominantFrequencyInBand:
    """Tests verifying that the dominant frequency lies within the passband."""

    def _dominant_freq(self, axis: np.ndarray, sample_rate: float = 100.0) -> float:
        """Return the frequency with the highest PSD."""
        freqs, psd = welch(axis, fs=sample_rate, nperseg=256)
        return float(freqs[np.argmax(psd)])

    def test_dominant_freq_z_in_band(self, denoised_signal: np.ndarray) -> None:
        """Dominant frequency of Z axis must be within [1, 10] Hz."""
        freq = self._dominant_freq(denoised_signal[0])
        assert 1.0 <= freq <= 10.0

    def test_dominant_freq_n_in_band(self, denoised_signal: np.ndarray) -> None:
        """Dominant frequency of N axis must be within [1, 10] Hz."""
        freq = self._dominant_freq(denoised_signal[1])
        assert 1.0 <= freq <= 10.0

    def test_dominant_freq_e_in_band(self, denoised_signal: np.ndarray) -> None:
        """Dominant frequency of E axis must be within [1, 10] Hz."""
        freq = self._dominant_freq(denoised_signal[2])
        assert 1.0 <= freq <= 10.0
