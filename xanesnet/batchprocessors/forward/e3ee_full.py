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

"""Batch processor for the E3EEFull dataset and model combination."""

import numpy as np
import torch

from xanesnet.datasets import E3EEFullBatch

from ..registry import BatchProcessorRegistry
from .base import ForwardBatchProcessor


@BatchProcessorRegistry.register(("e3ee_full", "e3ee_full"))
@BatchProcessorRegistry.register(("e3ee_full_mp", "e3ee_full"))
class E3EEFullBatchProcessor(ForwardBatchProcessor):
    """Batch processor for E3EEFull dataset + E3EEFull model.

    The model emits per-atom spectra ``(B, N_max, nE)``; this processor
    selects target-site rows via ``target_site_mask`` for loss computation
    (same pattern as SchNet/DimeNet).
    """

    def input_preparation(self, batch: E3EEFullBatch) -> dict[str, torch.Tensor]:
        """Prepare E3EEFull model inputs from the batch.

        Args:
            batch: Collated E3EEFull batch.

        Returns:
            Dict of tensors matching :meth:`xanesnet.models.e3ee_full.e3ee_full.E3EEFull.forward`.
        """
        return {
            "x": batch.x,
            "mask": batch.mask,
            "target_site_mask": batch.target_site_mask,
            "edge_src": batch.edge_src,
            "edge_dst": batch.edge_dst,
            "edge_weight": batch.edge_weight,
            "edge_vec": batch.edge_vec,
            "att_src": batch.att_src,
            "att_dst": batch.att_dst,
            "att_dist": batch.att_dist,
            "att_vec": batch.att_vec,
            "energies": batch.energies,
            "path_center": batch.path_center,
            "path_j": batch.path_j,
            "path_k": batch.path_k,
            "path_r0j": batch.path_r0j,
            "path_r0k": batch.path_r0k,
            "path_rjk": batch.path_rjk,
            "path_cosangle": batch.path_cosangle,
        }

    def prediction_preparation(self, batch: E3EEFullBatch, predictions: torch.Tensor) -> torch.Tensor:
        """Select target-site spectra from the padded per-atom output.

        Args:
            batch: Collated E3EEFull batch carrying ``target_site_mask``. ``(B, N_max)``
            predictions: Per-atom output tensor. ``(B, N_max, nE)``

        Returns:
            Spectra for target-site atoms only. ``(n_targets, nE)``
        """
        return predictions[batch.target_site_mask]

    def target_preparation(self, batch: E3EEFullBatch) -> torch.Tensor:
        """Prepare target spectra from an E3EEFull batch.

        Args:
            batch: Collated E3EEFull batch.

        Returns:
            Target spectra for target-site atoms only. ``(n_targets, n_energies)``
        """
        return batch.intensities

    def element_preparation(self, batch: E3EEFullBatch) -> torch.Tensor | None:
        """Prepare target-site atomic numbers from an E3EEFull batch.

        Selects atomic numbers at target-site positions via ``target_site_mask``.

        Args:
            batch: Collated E3EEFull batch.

        Returns:
            Target-site atomic numbers. ``(n_targets,)``
        """
        return batch.x[batch.target_site_mask]

    def sample_id_preparation(self, batch: E3EEFullBatch) -> np.ndarray:
        """Prepare file names from an E3EEFull batch.

        Args:
            batch: Collated E3EEFull batch.

        Returns:
            Array of file name strings. ``(n_target_sites,)``
        """
        return np.array(batch.sample_id, dtype=str)
