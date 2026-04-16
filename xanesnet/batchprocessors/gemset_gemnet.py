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

from xanesnet.datasets import GemNetBatch

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("gemset", "gemnet")
class GemsetGemNetBatchProcessor(BatchProcessor):

    def input_preparation(self, batch: GemNetBatch) -> dict[str, torch.Tensor | None]:
        inputs = {
            "z": batch.atomic_numbers,
            "pos": batch.atom_positions,
            "id_a": batch.id_a,
            "id_c": batch.id_c,
            "id_swap": batch.id_swap,
            "id3_expand_ba": batch.id3_expand_ba,
            "id3_reduce_ca": batch.id3_reduce_ca,
            "batch_seg": batch.batch_seg,
            "Kidx3": batch.Kidx3,
            # only if not triplets_only:
            "Kidx4": batch.Kidx4,
            "id4_int_b": batch.id4_int_b,
            "id4_int_a": batch.id4_int_a,
            "id4_reduce_ca": batch.id4_reduce_ca,
            "id4_reduce_cab": batch.id4_reduce_cab,
            "id4_expand_abd": batch.id4_expand_abd,
            "id4_reduce_intm_ca": batch.id4_reduce_intm_ca,
            "id4_expand_intm_db": batch.id4_expand_intm_db,
            "id4_reduce_intm_ab": batch.id4_reduce_intm_ab,
            "id4_expand_intm_ab": batch.id4_expand_intm_ab,
        }
        return inputs

    def target_preparation(self, batch: GemNetBatch) -> torch.Tensor:
        return batch.intensities

    def sample_id_extraction(self, batch: GemNetBatch) -> np.ndarray:
        return np.array(batch.sample_id, dtype=str)
