"""Shared pytest fixtures for all test suites."""

import numpy as np
import pytest


@pytest.fixture
def n_samples() -> int:
    """Return the default number of samples used across tests."""
    return 1000


@pytest.fixture
def sample_rate() -> float:
    """Return the default sample rate in Hz used across tests."""
    return 100.0


@pytest.fixture
def raw_signal(n_samples: int) -> np.ndarray:
    """Return a reproducible 3-axis signal of shape (3, N).

    Generated with seed=42 so every test run is deterministic.
    The signal includes a non-zero DC offset per axis to make
    baseline correction tests meaningful.

    Args:
        n_samples: Number of samples per axis (injected by pytest).

    Returns:
        Float64 array of shape (3, N).
    """
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((3, n_samples))
    offsets = np.array([[5.0], [-3.0], [1.5]])
    return noise + offsets


@pytest.fixture
def timestamps(n_samples: int, sample_rate: float) -> np.ndarray:
    """Return Unix timestamps corresponding to the raw_signal fixture.

    Args:
        n_samples: Number of samples (injected by pytest).
        sample_rate: Sampling frequency in Hz (injected by pytest).

    Returns:
        Float64 array of shape (N,) starting at t=0.
    """
    return np.arange(n_samples, dtype=np.float64) / sample_rate
