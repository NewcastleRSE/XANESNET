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

from xanesnet.datasets import E3EEBatch

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("e3ee", "e3ee")
class E3EEBatchProcessor(BatchProcessor):
    """
    Batch processor for the E3EE dataset + E3EE model combination.

    The E3EE dataset collate_fn stacks tensors into padded batches:
        x:           [B, N]    atomic numbers
        pos:         [B, N, 3] Cartesian coordinates
        mask:        [B, N]    boolean mask indicating valid atoms (True) vs padding (False)
        energies:    [B, nE]   energy grid
        intensities: [B, nE]   XANES intensities (targets)
    """

    def input_preparation(self, batch: E3EEBatch) -> dict[str, torch.Tensor]:
        return {
            "x": batch.x,
            "pos": batch.pos,
            "mask": batch.mask,
            "energies": batch.energies,
        }

    def target_preparation(self, batch: E3EEBatch) -> torch.Tensor:
        return batch.intensities

    def file_name_extraction(self, batch: E3EEBatch) -> np.ndarray:
        return np.array(batch.file_name, dtype=str)
