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

from .base import BatchProcessor
from .registry import BatchProcessorRegistry


@BatchProcessorRegistry.register("envembed", "envembed")
class EnvEmbedBatchProcessor(BatchProcessor):
    """
    Batch processor for the EnvEmbed dataset + EnvEmbed model combination.
    """

    def input_preparation(self, batch) -> dict[str, torch.Tensor]:
        # TODO implement input preparation
        raise NotImplementedError

    def target_preparation(self, batch) -> torch.Tensor:
        # TODO implement target preparation
        raise NotImplementedError

    def sample_id_extraction(self, batch) -> np.ndarray:
        return np.array(batch.sample_id, dtype=str)
