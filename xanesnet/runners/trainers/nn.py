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

from typing import Any

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model

from .base import Trainer
from .registry import TrainerRegistry


@TrainerRegistry.register("nntrainer")
class NNTrainer(Trainer):
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
        loss: dict[str, Any],
        regularizer: dict[str, Any],
        # trainer params:
        trainer_type: str,
        epochs: int,
        learning_rate: float,
        optimizer: str,
        lr_scheduler: dict[str, Any],
        early_stopper: dict[str, Any],
    ) -> None:
        super().__init__(
            dataset,
            model,
            device,
            batch_size,
            shuffle,
            drop_last,
            num_workers,
            loss,
            regularizer,
            trainer_type,
            epochs,
            learning_rate,
            optimizer,
            lr_scheduler,
            early_stopper,
        )
