#!/usr/bin/env bash
# run_pipeline.sh — convenience wrapper around the seismotrack CLI.
#
# Usage:
#   ./run_pipeline.sh [--output PATH] [OPTIONS]
#
# All arguments are forwarded directly to the CLI.
# Run with --help to see all available options.
#
# Example (defaults):
#   ./run_pipeline.sh --output data/output/signal.h5
#
# Example (custom):
#   ./run_pipeline.sh \
#     --output data/output/signal.h5 \
#     --n-samples 5000            \
#     --sample-rate 200.0         \
#     --f-low 1.0                 \
#     --f-high 20.0               \
#     --taper-alpha 0.1           \
#     --seed 42

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${SCRIPT_DIR}/venv/bin/python3" ]]; then
    echo "ERROR: virtualenv not found at ${SCRIPT_DIR}/venv" >&2
    echo "       Run: python3 -m venv venv && pip install -r requirements.txt" >&2
    exit 1
fi

exec "${SCRIPT_DIR}/venv/bin/python3" -m seismotrack.cli.cli "$@"
