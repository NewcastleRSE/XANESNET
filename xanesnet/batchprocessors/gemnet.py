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

"""Batch processor for the GemNet model."""

import numpy as np
import torch

from xanesnet.datasets import GemNetBatch

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("gemnet", "gemnet")
@BatchProcessorRegistry.register("gemnet_mp", "gemnet")
class GemNetBatchProcessor(BatchProcessor):
    """Batch processor for the PyG-based ``GemNetDataset`` feeding the GemNet model.

    All graph indices are precomputed in the dataset; this processor just forwards them
    and applies ``absorber_mask`` on the model's per-atom predictions.
    """

    def input_preparation(self, batch: GemNetBatch) -> dict[str, torch.Tensor | None]:
        """Prepare GemNet model inputs from the batch.

        Args:
            batch: Collated GemNet batch.

        Returns:
            Dict of tensors matching :meth:`xanesnet.models.gemnet.gemnet.GemNet.forward`.
        """
        inputs: dict[str, torch.Tensor | None] = {
            "z": batch.x,
            "edge_vec": batch.edge_vec,
            "edge_weight": batch.edge_weight,
            "id_c": batch.id_c,
            "id_a": batch.id_a,
            "id_swap": batch.id_swap,
            "id3_expand_ba": batch.id3_expand_ba,
            "id3_reduce_ca": batch.id3_reduce_ca,
            "Kidx3": batch.Kidx3,
        }
        # Quadruplet inputs (optional; present when dataset.quadruplets=True and model.triplets_only=False)
        for key in (
            "int_edge_vec",
            "int_edge_weight",
            "Kidx4",
            "id4_reduce_ca",
            "id4_reduce_cab",
            "id4_expand_abd",
            "id4_reduce_intm_ca",
            "id4_expand_intm_db",
            "id4_reduce_intm_ab",
            "id4_expand_intm_ab",
        ):
            inputs[key] = getattr(batch, key, None)
        return inputs

    def prediction_preparation(self, batch: GemNetBatch, predictions: torch.Tensor) -> torch.Tensor:
        """Select absorber-site predictions from the per-atom output.

        Args:
            batch: Collated GemNet batch carrying ``absorber_mask``.
            predictions: Per-atom output tensor. ``(num_atoms_total, num_targets)``

        Returns:
            Predictions for absorber atoms only. ``(n_abs, num_targets)``
        """
        return predictions[batch.absorber_mask]

    def target_preparation(self, batch: GemNetBatch) -> torch.Tensor:
        """Prepare target spectra from the batch.

        Args:
            batch: Collated GemNet batch.

        Returns:
            Target spectra for absorber atoms only. ``(n_abs, n_energies)``
        """
        return batch.intensities

    def file_name_extraction(self, batch: GemNetBatch) -> np.ndarray:
        """Extract file names from the batch.

        Args:
            batch: Collated GemNet batch.

        Returns:
            Array of file name strings aligned with absorber targets. ``(n_abs,)``
        """
        return np.array(batch.file_name, dtype=str)
