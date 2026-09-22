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

"""Symmetric-extension Fourier spectra encoding for XANESNET."""

import torch

from xanesnet.serialization.config import Config
from xanesnet.utils.math import fft, inverse_fft

from .base import SpectraEncoding
from .registry import SpectraEncodingRegistry


@SpectraEncodingRegistry.register("fourier")
class FourierEncoding(SpectraEncoding):
    """Symmetric-extension Fourier (DCT-like) spectra encoding.

    Encodes spectra with :func:`~xanesnet.utils.math.fft`, which builds an
    even-symmetric extension of the signal and takes the real part of its FFT,
    and decodes them with the matching
    :func:`~xanesnet.utils.math.inverse_fft`. For an input spectrum of length
    ``N`` the encoded dimension is ``2N`` (``concat=False``) or ``3N``
    (``concat=True``), where the leading ``N`` values are the original signal.

    Args:
        encoding_type: Identifier string for this encoding type.
        concat: Whether the encoded representation concatenates the original
            signal in front of the Fourier coefficients.
    """

    def __init__(
        self,
        encoding_type: str,
        concat: bool,
    ) -> None:
        """Initialize ``FourierEncoding``."""
        super().__init__(encoding_type)

        self.concat = concat

    def output_size(self, input_size: int) -> int:
        """Return the Fourier representation width for an input width.

        Args:
            input_size: Number of points in the input spectrum.

        Returns:
            ``2 * input_size`` or ``3 * input_size`` when ``concat`` is true.
        """
        return input_size * (3 if self.concat else 2)

    def encode(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the symmetric-extension FFT to target spectra.

        Args:
            targets: Ground-truth target spectra ``(B, N)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.
                Ignored: this encoding is element-independent.

        Returns:
            Fourier-encoded targets ``(B, 3N)`` when ``concat=True`` or
            ``(B, 2N)`` when ``concat=False``.
        """
        return fft(targets, self.concat)

    def decode(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Invert the symmetric-extension FFT applied by :meth:`encode`.

        Args:
            predictions: Model predictions in the Fourier space ``(B, 3N)`` when
                ``concat=True`` or ``(B, 2N)`` when ``concat=False``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.
                Ignored: this encoding is element-independent.

        Returns:
            Reconstructed spectra ``(B, N)``.
        """
        return inverse_fft(predictions, self.concat)

    @property
    def signature(self) -> list[Config]:
        """Return the Fourier-encoding signature.

        Returns:
            Configuration values needed to recreate this encoding.
        """
        sig = super().signature[0]
        sig.update_with_dict({"concat": self.concat})
        return [sig]
