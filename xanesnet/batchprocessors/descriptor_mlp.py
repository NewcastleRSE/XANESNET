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

"""Batch processor for the descriptor + MLP model combination."""

import numpy as np
import torch

from xanesnet.datasets import DescriptorData

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("descriptor", "mlp")
@BatchProcessorRegistry.register("descriptor_mp", "mlp")
class XanesXMLPBatchProcessor(BatchProcessor):
    """Batch processor for ``DescriptorData`` feeding the MLP model.

    Extracts the precomputed descriptor vector ``x`` as model input and the
    spectral intensity array ``y`` as the target.
    """

    def input_preparation(self, batch: DescriptorData) -> dict[str, torch.Tensor]:
        """Prepare MLP inputs from a descriptor batch.

        Args:
            batch: Collated descriptor batch.

        Returns:
            Dict with ``"x"`` containing the descriptor tensor. ``(batch_size, n_features)``

        Raises:
            ValueError: If ``batch.x`` is ``None``.
        """
        if batch.x is None:
            raise ValueError("Input data 'x' is None!")
        return {"x": batch.x}

    def target_preparation(self, batch: DescriptorData) -> torch.Tensor:
        """Prepare targets from a descriptor batch.

        Args:
            batch: Collated descriptor batch.

        Returns:
            Spectral intensity tensor. ``(batch_size, n_energies)``

        Raises:
            ValueError: If ``batch.y`` is ``None``.
        """
        if batch.y is None:
            raise ValueError("Target data 'y' is None!")
        return batch.y

    def file_name_extraction(self, batch: DescriptorData) -> np.ndarray:
        """Extract file names from a descriptor batch.

        Args:
            batch: Collated descriptor batch.

        Returns:
            Array of file name strings. ``(batch_size,)``

        """
        return np.array(batch.file_name, dtype=str)
