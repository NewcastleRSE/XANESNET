# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   ...

"""Gaussian-basis spectra encoding for XANESNET."""

import torch

from xanesnet.serialization.auto_config.registries import EncodingAutoResolver
from xanesnet.serialization.auto_config.statistics import SpectralStatisticsCollector
from xanesnet.serialization.config import Config, ConfigRaw
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.math import SpectralBasis, gaussian_fit, gaussian_inverse

from .base import SpectraEncoding
from .registry import SpectraEncodingRegistry


@SpectraEncodingRegistry.register("gaussian")
class GaussianEncoding(SpectraEncoding):
    """Gaussian-basis spectra encoding.

    Encodes spectra as coefficients of a Gaussian basis expansion and decodes
    them by re-synthesising spectra from those coefficients. The basis places
    Gaussians of the requested ``widths`` on a uniform grid of ``num_points``
    points, strided by ``basis_stride``. The encoded dimension is
    ``len(widths) * ceil(num_points / basis_stride)``.

    Widths and the stride are expressed in spectral grid points (bins) rather
    than energy units, so the encoding depends only on the number of points in
    the spectrum. ``num_points`` is dataset dependent and is typically resolved
    automatically from the prepared dataset via the encoding ``"auto"`` field.

    Args:
        encoding_type: Identifier string for this encoding type.
        widths: Gaussian standard deviations in spectral grid points (bins),
            one basis family per entry.
        basis_stride: Spacing between Gaussian centers in grid points.
        num_points: Number of points ``N`` in the spectra to encode.
        nonneg_output: Whether decoded spectra are clamped to non-negative
            values.

    Raises:
        ConfigError: If ``widths`` is empty, ``basis_stride`` is not positive,
            or ``num_points`` is not positive.
    """

    def __init__(
        self,
        encoding_type: str,
        widths: list[float],
        basis_stride: int,
        num_points: int,
        nonneg_output: bool,
    ) -> None:
        """Initialize ``GaussianEncoding``."""
        super().__init__(encoding_type)

        if not widths:
            raise ConfigError("GaussianEncoding requires at least one basis width.")
        if basis_stride <= 0:
            raise ConfigError("GaussianEncoding basis_stride must be positive.")
        if num_points <= 0:
            raise ConfigError("GaussianEncoding num_points must be positive.")

        self.widths = widths
        self.basis_stride = basis_stride
        self.num_points = num_points
        self.nonneg_output = nonneg_output

        # The basis is built on a unit grid (dE = 1), so widths are interpreted
        # in spectral grid points. Internal buffers are moved to the input
        # device lazily inside encode/decode.
        energies = torch.arange(num_points, dtype=torch.float32)
        self.basis = SpectralBasis(
            energies=energies,
            widths_eV=widths,
            normalize_atoms=True,
            stride=basis_stride,
        )

    def output_size(self, input_size: int) -> int:
        """Return the Gaussian coefficient width for an input spectrum.

        Args:
            input_size: Number of points in the input spectrum.

        Returns:
            Number of Gaussian basis coefficients.

        Raises:
            ValueError: If ``input_size`` does not match ``num_points`` used
                to construct the basis.
        """
        if input_size != self.num_points:
            raise ValueError(f"GaussianEncoding expected input width {self.num_points}, got {input_size}.")
        return len(self.widths) * ((input_size + self.basis_stride - 1) // self.basis_stride)

    def encode(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Fit Gaussian basis coefficients to target spectra.

        Args:
            targets: Ground-truth target spectra ``(B, N)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.
                Ignored: this encoding is element-independent.

        Returns:
            Gaussian basis coefficients ``(B, K)``.
        """
        self.basis.to(targets.device)
        return gaussian_fit(basis=self.basis, intensities=targets)

    def decode(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Re-synthesise spectra from predicted Gaussian basis coefficients.

        Args:
            predictions: Predicted Gaussian basis coefficients ``(B, K)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.
                Ignored: this encoding is element-independent.

        Returns:
            Reconstructed spectra ``(B, N)``.
        """
        self.basis.to(predictions.device)
        return gaussian_inverse(basis=self.basis, coeffs=predictions, nonneg_output=self.nonneg_output)

    @property
    def signature(self) -> list[Config]:
        """Return the Gaussian-encoding signature.

        Returns:
            Configuration values needed to recreate this encoding.
        """
        sig = super().signature[0]
        sig.update_with_dict(
            {
                "widths": self.widths,
                "basis_stride": self.basis_stride,
                "num_points": self.num_points,
                "nonneg_output": self.nonneg_output,
            }
        )
        return [sig]


@EncodingAutoResolver.register("gaussian")
def resolve_gaussian_encoding(item: ConfigRaw, statistics: SpectralStatisticsCollector) -> ConfigRaw:
    """Resolve the Gaussian-encoding spectral grid size.

    Args:
        item: Raw Gaussian encoding configuration dictionary.
        statistics: Streaming statistics of the training spectra
            (:class:`~xanesnet.serialization.auto_config.statistics.SpectralStatisticsCollector`).

    Returns:
        Mapping with Gaussian-encoding automatic field ``num_points``.
    """
    return {"num_points": statistics.overall.num_points}
