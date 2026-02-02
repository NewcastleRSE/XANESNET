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

import logging
from abc import ABC
from typing import Any

import torch

from xanesnet.batchprocessors import BatchProcessor, BatchProcessorRegistry
from xanesnet.datasets import Dataset
from xanesnet.losses import Loss, LossRegistry
from xanesnet.models import Model
from xanesnet.regularizers import Regularizer, RegularizerRegistry


class Runner(ABC):
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
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.device = device

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.loss_config = loss
        self.regularizer_config = regularizer

    def _setup_batchprocessor(self) -> BatchProcessor:
        batchprocessor = BatchProcessorRegistry.get(self.dataset.dataset_type, self.model.model_type)()
        return batchprocessor

    def _setup_dataloader(self) -> Any:
        dataloader_cls = self.dataset.get_dataloader()

        dataloader = dataloader_cls(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.dataset.collate_fn,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

        return dataloader

    def _setup_loss(self) -> Loss:
        loss_config = self.loss_config
        loss_type = loss_config["loss_type"]

        loss = LossRegistry.get(loss_type)(**loss_config)

        return loss

    def _setup_regularizer(self) -> Regularizer:
        regularizer_config = self.regularizer_config
        regularizer_type = regularizer_config["regularizer_type"]

        regularizer = RegularizerRegistry.get(regularizer_type)(**regularizer_config)

        return regularizer

    @staticmethod
    def _log_epoch_loss(
        loss: float,
        regularization: float,
        total: float,
        valid_loss: float | None = None,
        valid_regularization: float | None = None,
        valid_total: float | None = None,
        epoch: int | None = None,
    ) -> None:
        """
        Log training/validation/inference metrics for an epoch.
        """
        epoch_str = f"Epoch {epoch:03d} | " if epoch is not None else ""
        train_str = f"Loss: {loss:.6f} | Reg: {regularization:.6f} | Total: {total:.6f}"

        if valid_total is not None:
            valid_str = (
                f"Valid Loss: {valid_loss:.6f} | Valid Reg: {valid_regularization:.6f} | Valid Total: {valid_total:.6f}"
            )
            logging.info(f"{epoch_str}{train_str} | {valid_str}")
        else:
            logging.info(f"{epoch_str}{train_str}")
