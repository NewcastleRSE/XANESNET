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

from xanesnet.datasets import GeometryGraphBatch

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("geometrygraph", "schnet")
@BatchProcessorRegistry.register("geometrygraph_mp", "schnet")
class GeometryGraphSchNetBatchProcessor(BatchProcessor):

    def input_preparation(self, batch: GeometryGraphBatch) -> dict[str, torch.Tensor]:
        return {
            "z": batch.x,
            "edge_index": batch.edge_index,
            "edge_weight": batch.edge_weight,
            "batch": batch.batch,
        }

    def prediction_preparation(self, batch: GeometryGraphBatch, predictions: torch.Tensor) -> torch.Tensor:
        return predictions[batch.absorber_mask]

    def target_preparation(self, batch: GeometryGraphBatch) -> torch.Tensor:
        return batch.intensities

    def file_name_extraction(self, batch: GeometryGraphBatch) -> np.ndarray:
        return np.array(batch.file_name, dtype=str)
