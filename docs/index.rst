seismotrack
===========

A Python pipeline for processing raw seismic signals from 3-axis sensors
(Z, N, E). Applies a chain of denoising steps and exports cleaned data to
HDF5 format with scientific metadata.

Usage
-----

Install dependencies and run the pipeline::

   pip install -r requirements.txt
   ./run_pipeline.sh --output data/output/signal.h5

All options::

   ./run_pipeline.sh --help


API Reference
=============

Interfaces
----------

.. automodule:: seismotrack.interfaces.i_seismic_generator
   :members:

.. automodule:: seismotrack.interfaces.i_denoising_step
   :members:

.. automodule:: seismotrack.interfaces.i_exporter
   :members:

Ingestion
---------

.. automodule:: seismotrack.ingestion.seismic_signal_generator
   :members:

Denoising
---------

.. automodule:: seismotrack.denoising.baseline_step
   :members:

.. automodule:: seismotrack.denoising.taper_step
   :members:

.. automodule:: seismotrack.denoising.bandpass_step
   :members:

.. automodule:: seismotrack.denoising.denoiser
   :members:

Export
------

.. automodule:: seismotrack.export.hdf5_exporter
   :members:

Pipeline
--------

.. automodule:: seismotrack.pipeline.pipeline
   :members:

CLI
---

.. automodule:: seismotrack.cli.cli
   :members:
