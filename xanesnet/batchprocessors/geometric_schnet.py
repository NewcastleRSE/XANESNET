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
from torch_geometric.data import Data

from xanesnet.datasets import GeometricBatch

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("geometric", "schnet")
class GeometricSchNet(BatchProcessor):

    def input_preparation(self, batch: GeometricBatch) -> dict[str, torch.Tensor]:
        return {"z": batch.x, "pos": batch.pos, "batch": batch.batch}

    def input_preparation_single(self, sample: Data) -> dict[str, torch.Tensor]:
        assert sample.x is not None, "Input data 'x' is None!"
        assert sample.pos is not None, "Input data 'pos' is None!"
        return {"z": sample.x, "pos": sample.pos, "batch": torch.zeros(sample.x.size(0), dtype=torch.long)}

    def target_preparation(self, batch: GeometricBatch) -> torch.Tensor:
        return batch.intensities

    def target_preparation_single(self, sample: Data) -> torch.Tensor:
        return sample.intensities.unsqueeze(0)

    def sample_id_extraction(self, batch: GeometricBatch) -> np.ndarray:
        return np.array(batch.sample_id.cpu(), dtype=str)
