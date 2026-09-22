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

"""Direction-specific base class for forward XANESNET batch processors."""

from typing import Any

import torch

from ..base import BatchProcessor


class ForwardBatchProcessor(BatchProcessor):
    """Base class for forward batch processors (spectra are the model target).

    Forward prediction maps a structure representation (descriptors, graphs,
    ...) to a spectrum. The spectral encoding is therefore applied to the
    target and reversed on predictions, while inputs pass through unchanged.
    Concrete forward processors only implement the data-shaping methods
    (:meth:`input_preparation`, :meth:`target_preparation`,
    :meth:`sample_id_preparation`, and optionally :meth:`element_preparation`
    and :meth:`prediction_preparation`).
    """

    def encode_input(self, inputs: dict[str, Any], elements: torch.Tensor | None = None) -> dict[str, Any]:
        """Return inputs unchanged; forward inputs are not spectra.

        Args:
            inputs: Input dict returned by :meth:`input_preparation`.
            elements: Unused; present for interface compatibility.

        Returns:
            ``inputs`` unchanged.
        """
        return inputs

    def encode_target(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Encode the spectral target into the model's prediction space.

        Args:
            targets: Raw spectral target tensor.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.

        Returns:
            Encoded target when an encoding is configured, otherwise
            ``targets`` unchanged.
        """
        if self._encoding is not None:
            return self._encoding.encode(targets, elements)
        return targets

    def decode_target(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Decode spectral predictions back to the raw target space.

        Args:
            predictions: Model output tensor in the encoded prediction space.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.

        Returns:
            Decoded predictions when an encoding is configured, otherwise
            ``predictions`` unchanged.
        """
        if self._encoding is not None:
            return self._encoding.decode(predictions, elements)
        return predictions
