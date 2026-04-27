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

"""Batch processor for SchNet with geometry graphs."""

import numpy as np
import torch

from xanesnet.datasets import GeometryGraphBatch

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("geometrygraph", "schnet")
@BatchProcessorRegistry.register("geometrygraph_mp", "schnet")
class GeometryGraphSchNetBatchProcessor(BatchProcessor):
    """Batch processor for the ``GeometryGraphDataset`` feeding SchNet.

    Forwards the minimal set of graph tensors required by SchNet (no triplet
    indices needed).
    """

    def input_preparation(self, batch: GeometryGraphBatch) -> dict[str, torch.Tensor]:
        """Prepare SchNet model inputs from the batch.

        Args:
            batch: Collated geometry-graph batch.

        Returns:
            Dict of tensors matching :meth:`xanesnet.models.schnet.schnet.SchNet.forward`.
        """
        return {
            "z": batch.x,
            "edge_index": batch.edge_index,
            "edge_weight": batch.edge_weight,
            "batch": batch.batch,
        }

    def prediction_preparation(self, batch: GeometryGraphBatch, predictions: torch.Tensor) -> torch.Tensor:
        """Select absorber-site predictions from the per-atom output.

        Args:
            batch: Collated geometry graph batch carrying ``absorber_mask``.
            predictions: Per-atom output tensor. ``(num_atoms_total, num_targets)``

        Returns:
            Predictions for absorber atoms only. ``(n_abs, reduce_channels_2)``
        """
        return predictions[batch.absorber_mask]

    def target_preparation(self, batch: GeometryGraphBatch) -> torch.Tensor:
        """Prepare target spectra from the batch.

        Args:
            batch: Collated geometry-graph batch.

        Returns:
            Target spectra for absorber atoms only. ``(n_abs, n_energies)``
        """
        return batch.intensities

    def file_name_extraction(self, batch: GeometryGraphBatch) -> np.ndarray:
        """Extract file names from the batch.

        Args:
            batch: Collated geometry-graph batch.

        Returns:
            Array of file name strings aligned with absorber targets. ``(n_abs,)``
        """
        return np.array(batch.file_name, dtype=str)
