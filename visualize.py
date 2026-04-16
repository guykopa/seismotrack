"""Visualization script: compare raw vs denoised signal."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from seismotrack.ingestion.seismic_signal_generator import SeismicSignalGenerator
from seismotrack.denoising.denoiser import Denoiser
from seismotrack.denoising.baseline_step import BaselineStep
from seismotrack.denoising.taper_step import TaperStep
from seismotrack.denoising.bandpass_step import BandpassStep

# --- Generate and denoise ---
N, SR = 5000, 200.0
generator = SeismicSignalGenerator(n_samples=N, sample_rate=SR, seed=42)
raw, timestamps = generator.generate()

denoiser = Denoiser(steps=[
    BaselineStep(),
    TaperStep(alpha=0.1),
    BandpassStep(f_low=2.0, f_high=20.0, sample_rate=SR),
])
denoised = denoiser.denoise(raw)

axes_labels = ["Z", "N", "E"]
colors_raw = ["#e74c3c", "#e67e22", "#e74c3c"]
colors_den = ["#2ecc71", "#3498db", "#9b59b6"]

fig, axs = plt.subplots(3, 2, figsize=(14, 9))
fig.suptitle("seismotrack — Raw vs Denoised signal", fontsize=14, fontweight="bold")

for i, label in enumerate(axes_labels):
    # --- Time domain ---
    ax = axs[i, 0]
    ax.plot(timestamps, raw[i], color=colors_raw[i], alpha=0.6, linewidth=0.8, label="Raw")
    ax.plot(timestamps, denoised[i], color=colors_den[i], linewidth=1.0, label="Denoised")
    ax.set_title(f"Axis {label} — Time domain")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (nm/s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Frequency domain ---
    ax = axs[i, 1]
    freqs_r, psd_r = welch(raw[i], fs=SR, nperseg=256)
    freqs_d, psd_d = welch(denoised[i], fs=SR, nperseg=256)
    ax.semilogy(freqs_r, psd_r, color=colors_raw[i], alpha=0.6, linewidth=0.8, label="Raw")
    ax.semilogy(freqs_d, psd_d, color=colors_den[i], linewidth=1.0, label="Denoised")
    ax.axvspan(2.0, 20.0, alpha=0.08, color="green", label="Passband [2–20 Hz]")
    ax.set_title(f"Axis {label} — Frequency domain")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/output/signal_comparison.png", dpi=150)
print("Saved: data/output/signal_comparison.png")
plt.show()
