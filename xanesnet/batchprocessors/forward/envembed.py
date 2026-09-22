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

"""Batch processor for the EnvEmbed dataset and model combination."""

import numpy as np
import torch

from xanesnet.datasets import EnvEmbedData
from xanesnet.utils.math import SpectralBasis

from ..registry import BatchProcessorRegistry
from .base import ForwardBatchProcessor


@BatchProcessorRegistry.register(("envembed", "envembed"))
@BatchProcessorRegistry.register(("envembed_mp", "envembed"))
class EnvEmbedBatchProcessor(ForwardBatchProcessor):
    """Batch processor for EnvEmbed dataset + EnvEmbed model.

    Forwards padded descriptor features, distance features, sample lengths,
    and the shared :class:`~xanesnet.utils.math.SpectralBasis` object.
    """

    def input_preparation(self, batch: EnvEmbedData) -> dict[str, torch.Tensor | SpectralBasis]:
        """Prepare EnvEmbed model inputs from the batch.

        Args:
            batch: Collated EnvEmbed batch.

        Returns:
            Dict matching :meth:`xanesnet.models.envembed.envembed.EnvEmbed.forward`.
        """
        return {
            "descriptor_features": batch.descriptor_features,  # type: ignore[dict-item]
            "distance_features": batch.distance_features,  # type: ignore[dict-item]
            "lengths": batch.lengths,  # type: ignore[dict-item]
            "basis": batch.basis,  # type: ignore[dict-item]
        }

    def target_preparation(self, batch: EnvEmbedData) -> torch.Tensor:
        """Prepare target spectra from an EnvEmbed batch.

        Args:
            batch: Collated EnvEmbed batch.

        Returns:
            Target spectra tensor. ``(batch_size, n_energies)``.
        """
        return batch.intensities  # type: ignore[return-value]

    def element_preparation(self, batch: EnvEmbedData) -> torch.Tensor | None:
        """Prepare target-site atomic numbers from an EnvEmbed batch.

        Args:
            batch: Collated EnvEmbed batch.

        Returns:
            Per-sample target-site atomic numbers ``(batch_size,)``, or
            ``None`` if the dataset was built without element information.
        """
        return batch.element

    def sample_id_preparation(self, batch: EnvEmbedData) -> np.ndarray:
        """Prepare file names from an EnvEmbed batch.

        Args:
            batch: Collated EnvEmbed batch.

        Returns:
            Array of file name strings. ``(batch_size,)``.
        """
        return np.array(batch.sample_id, dtype=str)
