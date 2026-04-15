# ARCHITECTURE.md — seismotrack

## Pattern: Pipeline with explicit interfaces

seismotrack models signal processing as a chain of independent transformation
steps. Each step receives a signal, transforms it, and passes it to the next.
Steps are unaware of each other. The orchestrator (Denoiser) is unaware of
what each step does internally.

## Why this pattern

The problem is inherently sequential: a raw seismic signal must pass through
baseline correction, tapering, and bandpass filtering in a fixed order before
it can be exported. Each transformation maps directly to one class. The
architecture mirrors the physics of the problem.

## Data flow