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


@BatchProcessorRegistry.register("gemnet", "gemnet_oc")
class GemNetOCBatchProcessor(BatchProcessor):
    """
    Batch processor for the PyG-based ``GemNetDataset`` feeding GemNet-OC.

    The GemNet-OC model forwards a single ``batch`` kwarg and reads all
    pre-computed indices (main graph, a2ee2a/a2a/qint graphs, triplet and
    quadruplet indices) directly from the batch attributes. The absorber
    mask is applied to the per-atom predictions downstream.
    """

    def input_preparation(self, batch: GemNetBatch) -> dict[str, object]:
        return {"batch": batch}

    def prediction_preparation(self, batch: GemNetBatch, predictions: torch.Tensor) -> torch.Tensor:
        return predictions[batch.absorber_mask]

    def target_preparation(self, batch: GemNetBatch) -> torch.Tensor:
        return batch.intensities

    def file_name_extraction(self, batch: GemNetBatch) -> np.ndarray:
        return np.array(batch.file_name, dtype=str)
