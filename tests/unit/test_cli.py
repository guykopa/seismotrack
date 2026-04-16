"""Unit tests for CLI argument parsing and wiring — RED phase.

All tests are written before the implementation exists.
"""

import pytest

from seismotrack.cli.cli import build_parser, main


class TestBuildParser:
    """Tests verifying argument definitions and defaults."""

    def test_output_is_required(self) -> None:
        """--output must be a required argument."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_output_is_stored(self) -> None:
        """--output value must be stored in args.output."""
        parser = build_parser()
        args = parser.parse_args(["--output", "out.h5"])
        assert args.output == "out.h5"

    def test_default_n_samples(self) -> None:
        """--n-samples must default to 1000."""
        parser = build_parser()
        args = parser.parse_args(["--output", "out.h5"])
        assert args.n_samples == 1000

    def test_default_sample_rate(self) -> None:
        """--sample-rate must default to 100.0."""
        parser = build_parser()
        args = parser.parse_args(["--output", "out.h5"])
        assert args.sample_rate == 100.0

    def test_default_f_low(self) -> None:
        """--f-low must default to 1.0."""
        parser = build_parser()
        args = parser.parse_args(["--output", "out.h5"])
        assert args.f_low == 1.0

    def test_default_f_high(self) -> None:
        """--f-high must default to 10.0."""
        parser = build_parser()
        args = parser.parse_args(["--output", "out.h5"])
        assert args.f_high == 10.0

    def test_default_taper_alpha(self) -> None:
        """--taper-alpha must default to 0.1."""
        parser = build_parser()
        args = parser.parse_args(["--output", "out.h5"])
        assert args.taper_alpha == 0.1

    def test_default_seed(self) -> None:
        """--seed must default to 42."""
        parser = build_parser()
        args = parser.parse_args(["--output", "out.h5"])
        assert args.seed == 42

    def test_custom_values_parsed(self) -> None:
        """All custom values must be correctly parsed."""
        parser = build_parser()
        args = parser.parse_args([
            "--output", "my/signal.h5",
            "--n-samples", "500",
            "--sample-rate", "200.0",
            "--f-low", "2.0",
            "--f-high", "20.0",
            "--taper-alpha", "0.2",
            "--seed", "7",
        ])
        assert args.output == "my/signal.h5"
        assert args.n_samples == 500
        assert args.sample_rate == 200.0
        assert args.f_low == 2.0
        assert args.f_high == 20.0
        assert args.taper_alpha == 0.2
        assert args.seed == 7


class TestMain:
    """Tests verifying that main() runs the pipeline end-to-end."""

    def test_main_creates_output_file(self, tmp_path) -> None:
        """main() must create the HDF5 output file at the given path."""
        output = str(tmp_path / "result.h5")
        main(["--output", output])
        assert (tmp_path / "result.h5").exists()

    def test_main_with_custom_params(self, tmp_path) -> None:
        """main() must succeed with all custom parameters specified."""
        output = str(tmp_path / "result.h5")
        main([
            "--output", output,
            "--n-samples", "512",
            "--sample-rate", "100.0",
            "--f-low", "1.0",
            "--f-high", "10.0",
            "--taper-alpha", "0.1",
            "--seed", "0",
        ])
        assert (tmp_path / "result.h5").exists()
