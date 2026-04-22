"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import torch

from xanesnet.datasets import E3EEFullBatch

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("e3ee_full", "e3ee_full")
class E3EEFullBatchProcessor(BatchProcessor):
    """
    Batch processor for the E3EEFull dataset + E3EEFull model combination.

    The model emits spectra for every atom in the padded layout
    (``[B, N_max, nE]``); this processor selects only the rows flagged by
    ``absorber_mask`` for loss computation (identical pattern to
    SchNet / DimeNet).
    """

    def input_preparation(self, batch: E3EEFullBatch) -> dict[str, torch.Tensor]:
        return {
            "x": batch.x,
            "mask": batch.mask,
            "edge_src": batch.edge_src,
            "edge_dst": batch.edge_dst,
            "edge_weight": batch.edge_weight,
            "edge_vec": batch.edge_vec,
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
        # predictions: [B, N_max, nE]; absorber_mask: [B, N_max] -> [n_abs, nE]
        return predictions[batch.absorber_mask]

    def target_preparation(self, batch: E3EEFullBatch) -> torch.Tensor:
        return batch.intensities

    def file_name_extraction(self, batch: E3EEFullBatch) -> np.ndarray:
        return np.array(batch.file_name, dtype=str)
