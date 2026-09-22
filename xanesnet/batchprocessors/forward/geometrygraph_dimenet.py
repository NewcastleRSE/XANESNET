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

"""Batch processor for DimeNet/DimeNet++ with geometry graphs."""

import numpy as np
import torch

from xanesnet.datasets import GeometryGraphBatch

from ..registry import BatchProcessorRegistry
from .base import ForwardBatchProcessor


@BatchProcessorRegistry.register(("geometrygraph", "dimenet"))
@BatchProcessorRegistry.register(("geometrygraph", "dimenet++"))
@BatchProcessorRegistry.register(("geometrygraph_mp", "dimenet"))
@BatchProcessorRegistry.register(("geometrygraph_mp", "dimenet++"))
class GeometryGraphDimeNetBatchProcessor(ForwardBatchProcessor):
    """Batch processor for ``GeometryGraphDataset`` + DimeNet/DimeNet++.

    Forwards standard geometry-graph tensors plus precomputed triplet
    interaction indices (``idx_kj``, ``idx_ji``). Target-site predictions
    are selected via ``target_site_mask``.
    """

    def input_preparation(self, batch: GeometryGraphBatch) -> dict[str, torch.Tensor]:
        """Prepare DimeNet model inputs from the batch.

        Args:
            batch: Collated geometry-graph batch.

        Returns:
            Dict of tensors matching :meth:`xanesnet.models.dimenet.dimenet.DimeNet.forward`.
        """
        return {
            "z": batch.x,
            "edge_index": batch.edge_index,
            "edge_weight": batch.edge_weight,
            "angle": batch.angle,
            "idx_kj": batch.idx_kj,
            "idx_ji": batch.idx_ji,
        }

    def prediction_preparation(self, batch: GeometryGraphBatch, predictions: torch.Tensor) -> torch.Tensor:
        """Select target-site predictions from the per-atom output.

        Args:
            batch: Collated geometry graph batch carrying ``target_site_mask``.
            predictions: Per-atom output tensor. ``(num_atoms_total, num_targets)``

        Returns:
            Predictions for target sites only. ``(n_target_sites, out_channels)``
        """
        return predictions[batch.target_site_mask]

    def target_preparation(self, batch: GeometryGraphBatch) -> torch.Tensor:
        """Prepare target spectra from a geometry-graph batch.

        Args:
            batch: Collated geometry-graph batch.

        Returns:
            Target spectra for target sites only. ``(n_target_sites, n_energies)``
        """
        return batch.intensities

    def element_preparation(self, batch: GeometryGraphBatch) -> torch.Tensor | None:
        """Prepare target-site atomic numbers from a geometry-graph batch.

        Selects atomic numbers at target-site positions via ``target_site_mask``.

        Args:
            batch: Collated geometry-graph batch.

        Returns:
            Target-site atomic numbers. ``(n_target_sites,)``
        """
        return batch.x[batch.target_site_mask]

    def sample_id_preparation(self, batch: GeometryGraphBatch) -> np.ndarray:
        """Prepare file names from a geometry-graph batch.

        Args:
            batch: Collated geometry-graph batch.

        Returns:
            Array of file name strings. ``(n_target_sites,)``
        """
        return np.array(batch.sample_id, dtype=str)
