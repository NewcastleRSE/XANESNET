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

"""Batch processor for the GemNet-OC model."""

import numpy as np
import torch

from xanesnet.datasets import GemNetBatch

from ..registry import BatchProcessorRegistry
from .base import ForwardBatchProcessor


@BatchProcessorRegistry.register(("gemnet_oc", "gemnet_oc"))
@BatchProcessorRegistry.register(("gemnet_oc_mp", "gemnet_oc"))
class GemNetOCBatchProcessor(ForwardBatchProcessor):
    """Batch processor for ``GemNetDataset`` + GemNet-OC.

    Forwards a large set of precomputed tensors (main graph, triplets,
    quadruplets, a2ee2a, a2a, mixed triplets). Optional fields are fetched
    with ``getattr`` so downstream model toggles can gate their use.
    """

    _OPTIONAL_KEYS: tuple[str, ...] = (
        # Quadruplet graph / indices
        "qint_edge_index",
        "qint_edge_weight",
        "qint_edge_vec",
        "id4_expand_intm_db",
        "id4_expand_intm_ab",
        "id4_reduce_intm_ab",
        "id4_reduce_intm_ca",
        "id4_reduce_ca",
        "id4_expand_abd",
        "id4_reduce_cab",
        "Kidx4",
        # a2ee2a graph
        "a2ee2a_edge_index",
        "a2ee2a_edge_weight",
        "a2ee2a_edge_vec",
        # Mixed triplets
        "trip_a2e_in",
        "trip_a2e_out",
        "trip_a2e_out_agg",
        "trip_e2a_in",
        "trip_e2a_out",
        "trip_e2a_out_agg",
        # a2a graph
        "a2a_edge_index",
        "a2a_edge_weight",
        "a2a_edge_vec",
    )

    def input_preparation(self, batch: GemNetBatch) -> dict[str, torch.Tensor | None]:
        """Prepare GemNet-OC model inputs from the batch.

        Args:
            batch: Collated GemNet-OC batch.

        Returns:
            Dict of tensors matching :meth:`xanesnet.models.gemnet_oc.gemnet_oc.GemNetOC.forward`.
        """
        inputs: dict[str, torch.Tensor | None] = {
            "z": batch.x,
            "edge_index": batch.edge_index,
            "edge_weight": batch.edge_weight,
            "edge_vec": batch.edge_vec,
            "id_swap": batch.id_swap,
            "id3_expand_ba": batch.id3_expand_ba,
            "id3_reduce_ca": batch.id3_reduce_ca,
            "Kidx3": batch.Kidx3,
        }
        for key in self._OPTIONAL_KEYS:
            inputs[key] = getattr(batch, key, None)
        return inputs

    def prediction_preparation(self, batch: GemNetBatch, predictions: torch.Tensor) -> torch.Tensor:
        """Select target-site predictions from the per-atom output.

        Args:
            batch: Collated GemNet batch carrying ``target_site_mask``.
            predictions: Per-atom output tensor. ``(num_atoms_total, num_targets)``

        Returns:
            Predictions for target sites only. ``(n_target_sites, num_targets)``
        """
        return predictions[batch.target_site_mask]

    def target_preparation(self, batch: GemNetBatch) -> torch.Tensor:
        """Prepare target spectra from a GemNet-OC batch.

        Args:
            batch: Collated GemNet-OC batch.

        Returns:
            Target spectra for target sites only. ``(n_target_sites, n_energies)``
        """
        return batch.intensities

    def element_preparation(self, batch: GemNetBatch) -> torch.Tensor | None:
        """Prepare target-site atomic numbers from a GemNet-OC batch.

        Selects atomic numbers at target-site positions via ``target_site_mask``.

        Args:
            batch: Collated GemNet-OC batch.

        Returns:
            Target-site atomic numbers. ``(n_target_sites,)``
        """
        return batch.x[batch.target_site_mask]

    def sample_id_preparation(self, batch: GemNetBatch) -> np.ndarray:
        """Prepare file names from a GemNet-OC batch.

        Args:
            batch: Collated GemNet-OC batch.

        Returns:
            Array of file name strings. ``(n_target_sites,)``
        """
        return np.array(batch.sample_id, dtype=str)
