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
        params: dict[str, Any],
        dataset: Dataset,
        model: Model,
        device: str | torch.device,
    ) -> None:
        self.params = params
        self.dataset = dataset
        self.model = model
        self.device = device

    def _setup_batchprocessor(self) -> BatchProcessor:
        batchprocessor = BatchProcessorRegistry.get(self.dataset.dataset_type, self.model.model_type)()
        return batchprocessor

    def _setup_dataloader(self) -> Any:
        dataloader_cls = self.dataset.get_dataloader()

        dataloader = dataloader_cls(
            self.dataset,
            batch_size=self.params["batch_size"],
            shuffle=self.params["shuffle"],
            collate_fn=self.dataset.collate_fn,
            drop_last=self.params["drop_last"],
            num_workers=self.params["num_workers"],
        )

        return dataloader

    def _setup_loss(self) -> Loss:
        loss_config = self.params["loss"]
        loss_type = loss_config["loss_type"]

        loss = LossRegistry.get(loss_type)(**loss_config)

        return loss

    def _setup_regularizer(self) -> Regularizer:
        regularizer_config = self.params["regularizer"]
        regularizer_type = regularizer_config["regularizer_type"]
        regularizer_params = regularizer_config["params"]
        regularizer = RegularizerRegistry.get(regularizer_type)(regularizer_type, **regularizer_params)

        return regularizer

    @staticmethod
    def _log_epoch_loss(
        loss: float,
        regularization: float,
        total: float,
        valid_loss: float | None = None,
        epoch: int | None = None,
    ) -> None:
        epoch_part = f"Epoch {epoch + 1:03d} | " if epoch is not None else ""
        validation_part = f"Validation: {valid_loss:.6f}" if valid_loss is not None else ""

        logging.info(
            f"{epoch_part}"
            f"Loss: {loss:.6f} | "
            f"Regularization: {regularization:.6f} | "
            f"Total: {total:.6f} | "
            f"{validation_part}"
        )
