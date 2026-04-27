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

"""Batch processor for the E3EE dataset and model combination."""

import numpy as np
import torch

from xanesnet.datasets import E3EEBatch

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("e3ee", "e3ee")
@BatchProcessorRegistry.register("e3ee_mp", "e3ee")
class E3EEBatchProcessor(BatchProcessor):
    """Batch processor for the E3EE dataset + E3EE model combination.

    Node features are padded to ``(B, N_max, ...)``; edge and absorber-path
    tensors are flat and carry indices into the padded ``B * N_max`` layout
    (already offset by the dataset collate_fn).
    """

    def input_preparation(self, batch: E3EEBatch) -> dict[str, torch.Tensor]:
        """Prepare E3EE model inputs from the batch.

        Args:
            batch: Collated E3EE batch.

        Returns:
            Dict of tensors matching :meth:`xanesnet.models.e3ee.e3ee.E3EE.forward`.
        """
        return {
            "x": batch.x,
            "mask": batch.mask,
            "absorber_index": batch.absorber_index,
            "edge_src": batch.edge_src,
            "edge_dst": batch.edge_dst,
            "edge_weight": batch.edge_weight,
            "edge_vec": batch.edge_vec,
            "att_dst": batch.att_dst,
            "att_dist": batch.att_dist,
            "att_vec": batch.att_vec,
            "energies": batch.energies,
            "path_j": batch.path_j,
            "path_k": batch.path_k,
            "path_r0j": batch.path_r0j,
            "path_r0k": batch.path_r0k,
            "path_rjk": batch.path_rjk,
            "path_cosangle": batch.path_cosangle,
            "path_batch": batch.path_batch,
        }

    def target_preparation(self, batch: E3EEBatch) -> torch.Tensor:
        """Prepare target spectra from the batch.

        Args:
            batch: Collated E3EE batch.

        Returns:
            Target spectra tensor. ``(batch_size, n_energies)``
        """
        return batch.intensities

    def file_name_extraction(self, batch: E3EEBatch) -> np.ndarray:
        """Extract file names from the batch.

        Args:
            batch: Collated E3EE batch.

        Returns:
            Array of file name strings. ``(batch_size,)``
        """
        return np.array(batch.file_name, dtype=str)
