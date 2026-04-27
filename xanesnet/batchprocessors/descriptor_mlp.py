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

from xanesnet.datasets import DescriptorData

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("descriptor", "mlp")
@BatchProcessorRegistry.register("descriptor_mp", "mlp")
class XanesXMLPBatchProcessor(BatchProcessor):

    def input_preparation(self, batch: DescriptorData) -> dict[str, torch.Tensor]:
        if batch.x is None:
            raise ValueError("Input data 'x' is None!")
        return {"x": batch.x}

    def target_preparation(self, batch: DescriptorData) -> torch.Tensor:
        if batch.y is None:
            raise ValueError("Target data 'y' is None!")
        return batch.y

    def file_name_extraction(self, batch: DescriptorData) -> np.ndarray:
        return np.array(batch.file_name, dtype=str)
