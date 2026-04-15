# CLAUDE.md — seismotrack

## Project overview
seismotrack is a Python pipeline for processing raw seismic signals from
3-axis sensors (Z, N, E). It applies a chain of denoising steps and exports
cleaned data to HDF5 format with scientific metadata.

## Architecture
Pipeline Pattern with explicit interfaces. See ARCHITECTURE.md for full details.

## Core principles
- TDD strictly enforced: tests are written before implementation, no exceptions
- SOLID principles applied at every level
- All code comments and docstrings in English
- Google-style docstrings on every public class and method

## TDD cycle — mandatory
For every new class or function:
1. RED: write a failing test in tests/unit/ or tests/integration/
2. GREEN: write the minimum code to make it pass
3. REFACTOR: clean without breaking tests
Never write implementation code before the corresponding test exists.

## SOLID mapping
- S: one file = one responsibility, never mix concerns
- O: add a new step by creating a new file, never modify existing steps
- L: every IDenoisingStep implementation must be substitutable
- I: ISeismicGenerator, IDenoisingStep, IExporter are separate interfaces
- D: Denoiser and Pipeline receive dependencies by injection, never instantiate concrete classes internally

## Project structure
seismotrack/
├── seismotrack/
│   ├── interfaces/         # abstract base classes only
│   ├── ingestion/          # signal generation only
│   ├── denoising/          # one file per step + orchestrator
│   ├── export/             # HDF5 export only
│   ├── pipeline/           # orchestration only
│   └── cli/                # argparse CLI only
├── tests/
│   ├── conftest.py         # shared fixtures
│   ├── unit/               # one test file per class
│   ├── integration/        # end-to-end pipeline tests
│   └── non_regression/     # signal stability tests
├── docs/                   # Sphinx autodoc
└── .github/workflows/      # CI: flake8 + pytest --cov

## Stack
- Python 3.11+
- numpy, scipy: signal processing
- h5py: HDF5 export
- pytest, pytest-cov: testing
- flake8: linting
- sphinx: documentation

## Naming conventions
- Interfaces: prefix I → IDenosingStep, IExporter
- Implementations: descriptive noun → BandpassStep, HDF5Exporter
- Tests: test_{class_name}.py → test_bandpass_step.py
- Fixtures: defined in conftest.py, never duplicated across test files

## Signal format
- shape (3, N): axis 0 = Z, axis 1 = N, axis 2 = E
- timestamps: 1D array of Unix float64 seconds, shape (N,)
- units: velocity in nm/s, frequency in Hz

## What to build — ordered task list
Build in this exact order, respecting TDD at each step:

1. seismotrack/interfaces/i_denoising_step.py
2. seismotrack/interfaces/i_seismic_generator.py
3. seismotrack/interfaces/i_exporter.py
4. tests/conftest.py
5. tests/unit/test_seismic_signal_generator.py → seismotrack/ingestion/seismic_signal_generator.py
6. tests/unit/test_baseline_step.py → seismotrack/denoising/baseline_step.py
7. tests/unit/test_taper_step.py → seismotrack/denoising/taper_step.py
8. tests/unit/test_bandpass_step.py → seismotrack/denoising/bandpass_step.py
9. tests/unit/test_denoiser.py → seismotrack/denoising/denoiser.py
10. tests/integration/test_hdf5_exporter.py → seismotrack/export/hdf5_exporter.py
11. tests/integration/test_pipeline.py → seismotrack/pipeline/pipeline.py
12. tests/non_regression/test_signal_stability.py
13. seismotrack/cli/cli.py
14. run_pipeline.sh
15. .github/workflows/ci.yml
16. docs/conf.py + docs/index.rst
17. README.md with CI badge, install instructions, usage example

## Rules Claude must follow
- Never write implementation before the test
- Never use print() for debugging, use logging
- Never hardcode paths, always use argparse or config
- Never catch bare Exception, always catch specific exceptions
- Always type-hint every function parameter and return value
- Always run the full test suite mentally before declaring a task done
- If a class has more than one reason to change, split it
- If a test requires more than 10 lines of setup, extract a fixture
