# seismotrack

[![CI](https://github.com/fkopa/seismotrack/actions/workflows/ci.yml/badge.svg)](https://github.com/fkopa/seismotrack/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](.coverage)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)

A Python pipeline for processing raw seismic signals from 3-axis sensors
(Z, N, E). Applies a chain of denoising steps and exports cleaned data to
HDF5 format with scientific metadata.

---

## Why seismotrack?

Every day, seismograph networks around the world record thousands of signals.
These signals are noisy: sensors drift, electromagnetic interference pollutes
the recording, and mechanical vibrations from nearby traffic or industrial
activity mask the faint signature of real seismic events.

Before any scientific analysis can begin — before a seismologist can locate
an earthquake, measure its magnitude, or study the structure of the Earth's
crust — the raw signal must be cleaned. This cleaning process is not optional.
A signal with a DC offset fed into a frequency filter produces violent
artefacts. A signal with sharp edges fed into a spectral analyser produces
false frequency peaks. The physics of the problem imposes a strict order of
operations.

seismotrack automates this pipeline:

```
Raw 3-axis signal (Z, N, E)
        │
        ▼
┌───────────────────┐
│   BaselineStep    │  Removes DC offset — centres the signal on zero
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    TaperStep      │  Fades edges to zero — prevents filter artefacts
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  BandpassStep     │  Keeps only useful frequencies (e.g. 1–10 Hz)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   HDF5Exporter    │  Writes signal + metadata to a portable HDF5 file
└───────────────────┘
```

The output is a clean, annotated HDF5 file ready for scientific analysis:
earthquake location algorithms, seismic tomography, early warning systems,
or structural health monitoring of buildings and bridges.

---

## Installation

**Requirements:** Python 3.11 or 3.12

```bash
git clone https://github.com/fkopa/seismotrack.git
cd seismotrack
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Quick start

```bash
./run_pipeline.sh --output data/output/signal.h5
```

### All options

```bash
./run_pipeline.sh \
  --output        data/output/signal.h5 \   # destination file (required)
  --n-samples     1000                   \   # number of samples per axis
  --sample-rate   100.0                  \   # sampling frequency in Hz
  --f-low         1.0                    \   # bandpass lower cutoff in Hz
  --f-high        10.0                   \   # bandpass upper cutoff in Hz
  --taper-alpha   0.1                    \   # Tukey window shape (0–1)
  --seed          42                         # random seed for reproducibility
```

### Reading the output

```python
import h5py
import numpy as np

with h5py.File("data/output/signal.h5", "r") as f:
    signal     = f["signal"][:]      # shape (3, N), units nm/s, axes Z/N/E
    timestamps = f["timestamps"][:]  # shape (N,),   units s (Unix)

print(signal.shape)     # (3, 1000)
print(timestamps[-1])   # 9.99
```

---

## Running the tests

```bash
# Full test suite with coverage
pytest --cov=seismotrack --cov-report=term-missing

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Non-regression (signal stability) tests
pytest tests/non_regression/
```

---

## Architecture

seismotrack follows the **Pipeline Pattern** with explicit interfaces.
Each component depends on an abstraction, never on a concrete class
(Dependency Inversion). Adding a new denoising step requires only creating
a new file — no existing code is modified (Open/Closed).

```
seismotrack/
├── interfaces/    # Abstract contracts: IDenoisingStep, IExporter, ISeismicGenerator
├── ingestion/     # Signal generation
├── denoising/     # One file per step + Denoiser orchestrator
├── export/        # HDF5 persistence
├── pipeline/      # Final assembly
└── cli/           # argparse entry point
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

---

## Project status

| Step | Status |
|---|---|
| Interfaces | done |
| SeismicSignalGenerator | done |
| BaselineStep | done |
| TaperStep | done |
| BandpassStep | done |
| Denoiser | done |
| HDF5Exporter | done |
| Pipeline | done |
| CLI | done |
| CI / GitHub Actions | done |
| Sphinx documentation | done |
