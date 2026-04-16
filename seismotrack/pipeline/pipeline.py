"""Pipeline orchestrator."""

from seismotrack.interfaces.i_seismic_generator import ISeismicGenerator
from seismotrack.interfaces.i_exporter import IExporter
from seismotrack.denoising.denoiser import Denoiser


class Pipeline:
    """Assemble and run the full seismic processing chain.

    Wires a generator, a denoiser, and an exporter together.
    Receives all dependencies by injection — never instantiates
    concrete classes internally.

    Args:
        generator: Produces the raw seismic signal and timestamps.
        denoiser: Applies the ordered denoising steps.
        exporter: Persists the processed signal to disk.
        output_path: Destination file path passed to the exporter.
    """

    def __init__(
        self,
        generator: ISeismicGenerator,
        denoiser: Denoiser,
        exporter: IExporter,
        output_path: str,
    ) -> None:
        self._generator = generator
        self._denoiser = denoiser
        self._exporter = exporter
        self._output_path = output_path

    def run(self) -> None:
        """Execute the full pipeline: generate, denoise, export.

        Generates a signal, passes it through the denoiser, then
        exports the result to the configured output path.
        """
        signal, timestamps = self._generator.generate()
        denoised = self._denoiser.denoise(signal)
        self._exporter.export(denoised, timestamps, self._output_path)
