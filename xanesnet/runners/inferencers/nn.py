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

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model

from .base import Inferencer
from .registry import InferencerRegistry


@InferencerRegistry.register("nninferencer")
class NNInferencer(Inferencer):
    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        device: str | torch.device,
        # runner params:
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        # inferencer params:
        inferencer_type: str,
    ) -> None:
        super().__init__(
            dataset,
            model,
            device,
            batch_size,
            shuffle,
            drop_last,
            num_workers,
            inferencer_type,
        )
