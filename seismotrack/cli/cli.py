"""Command-line interface for seismotrack."""

import argparse
import logging
from typing import List, Optional

from seismotrack.denoising.bandpass_step import BandpassStep
from seismotrack.denoising.baseline_step import BaselineStep
from seismotrack.denoising.denoiser import Denoiser
from seismotrack.denoising.taper_step import TaperStep
from seismotrack.export.hdf5_exporter import HDF5Exporter
from seismotrack.ingestion.seismic_signal_generator import SeismicSignalGenerator
from seismotrack.pipeline.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Process and denoise a synthetic seismic signal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination HDF5 file path.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of samples to generate per axis.",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=100.0,
        help="Sampling frequency in Hz.",
    )
    parser.add_argument(
        "--f-low", type=float, default=1.0,
        help="Lower cutoff frequency for the bandpass filter in Hz.",
    )
    parser.add_argument(
        "--f-high", type=float, default=10.0,
        help="Upper cutoff frequency for the bandpass filter in Hz.",
    )
    parser.add_argument(
        "--taper-alpha", type=float, default=0.1,
        help="Tukey window alpha parameter (0=rectangular, 1=Hann).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible signal generation.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments, build and run the pipeline.

    Args:
        argv: Argument list. Defaults to sys.argv when None.
    """
    args = build_parser().parse_args(argv)

    logger.info(
        "Starting pipeline: n_samples=%d sample_rate=%.1f Hz "
        "f_low=%.1f Hz f_high=%.1f Hz output=%s",
        args.n_samples, args.sample_rate,
        args.f_low, args.f_high, args.output,
    )

    pipeline = Pipeline(
        generator=SeismicSignalGenerator(
            n_samples=args.n_samples,
            sample_rate=args.sample_rate,
            seed=args.seed,
        ),
        denoiser=Denoiser(steps=[
            BaselineStep(),
            TaperStep(alpha=args.taper_alpha),
            BandpassStep(
                f_low=args.f_low,
                f_high=args.f_high,
                sample_rate=args.sample_rate,
            ),
        ]),
        exporter=HDF5Exporter(),
        output_path=args.output,
    )

    pipeline.run()
    logger.info("Pipeline complete. Output written to %s", args.output)


if __name__ == "__main__":
    main()
