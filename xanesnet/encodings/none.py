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

"""Identity (no-op) spectra encoding for XANESNET."""

import torch

from xanesnet.serialization.config import Config

from .base import SpectraEncoding
from .registry import SpectraEncodingRegistry


@SpectraEncodingRegistry.register("identity")
@SpectraEncodingRegistry.register("none")
class NoneEncoding(SpectraEncoding):
    """Identity spectra encoding.

    Leaves spectra unchanged in both directions. Use this as the explicit
    "no encoding" option when training should operate directly on raw spectra.

    Args:
        encoding_type: Identifier string for this encoding type.
    """

    def __init__(
        self,
        encoding_type: str,
    ) -> None:
        """Initialize ``NoneEncoding``."""
        super().__init__(encoding_type)

    def output_size(self, input_size: int) -> int:
        """Return the unchanged width of the identity encoding.

        Args:
            input_size: Number of points in the input spectrum.

        Returns:
            Number of points in the encoded representation.
        """
        return input_size

    def encode(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Return the target spectra unchanged.

        Args:
            targets: Ground-truth target spectra ``(B, N)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.
                Ignored: this encoding is element-independent.

        Returns:
            The input tensor unchanged.
        """
        return targets

    def decode(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Return the predictions unchanged.

        Args:
            predictions: Model predictions ``(B, N)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.
                Ignored: this encoding is element-independent.

        Returns:
            The input tensor unchanged.
        """
        return predictions

    @property
    def signature(self) -> list[Config]:
        """Return the identity-encoding signature.

        Returns:
            Configuration values needed to recreate this encoding.
        """
        return super().signature
