# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Batch processor for the EnvEmbed dataset and model combination."""

import numpy as np
import torch

from xanesnet.datasets import EnvEmbedData
from xanesnet.utils.math import SpectralBasis

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("envembed", "envembed")
@BatchProcessorRegistry.register("envembed_mp", "envembed")
class EnvEmbedBatchProcessor(BatchProcessor):
    """Batch processor for the EnvEmbed dataset + EnvEmbed model combination.

    The EnvEmbed dataset collate_fn produces an ``EnvEmbedData`` batch with:

    - ``descriptor_features``: ``(B, N, H)`` padded descriptor features
    - ``distance_features``: ``(B, N)`` distances from absorber
    - ``lengths``: ``(B,)`` number of real atoms per sample
    - ``file_name``: ``list[str]`` sample identifiers
    - ``basis``: ``SpectralBasis`` spectral basis object (not a tensor, not collated)
    """

    def input_preparation(self, batch: EnvEmbedData) -> dict[str, torch.Tensor | SpectralBasis]:
        """Prepare EnvEmbed model inputs from the batch.

        Args:
            batch: Collated EnvEmbed batch.

        Returns:
            Dict matching :meth:`xanesnet.models.envembed.envembed.EnvEmbed.forward`.

        Raises:
            ValueError: If any required model input is missing from the batch.
        """
        return {
            "descriptor_features": batch.descriptor_features,  # type: ignore
            "distance_features": batch.distance_features,  # type: ignore
            "lengths": batch.lengths,  # type: ignore
            "basis": batch.basis,  # type: ignore[dict-item]
        }

    def target_preparation(self, batch: EnvEmbedData) -> torch.Tensor:
        """Prepare target spectra from the batch.

        Args:
            batch: Collated EnvEmbed batch.

        Returns:
            Target spectra tensor. ``(batch_size, n_energies)``

        Raises:
            ValueError: If ``batch.intensities`` is ``None``.
        """
        return batch.intensities  # type: ignore type

    def file_name_extraction(self, batch: EnvEmbedData) -> np.ndarray:
        """Extract file names from the batch.

        Args:
            batch: Collated EnvEmbed batch.

        Returns:
            Array of file name strings. ``(batch_size,)``

        Raises:
            ValueError: If ``batch.file_name`` is ``None``.
        """
        return np.array(batch.file_name, dtype=str)  # type: ignore type
