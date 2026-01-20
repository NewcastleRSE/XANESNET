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
from xanesnet.registry import OptimizerRegistry
from xanesnet.regularizers import Regularizer, RegularizerRegistry


class Trainer(ABC):
    def __init__(
        self,
        trainer_type: str,
        params: dict[str, Any],
        dataset: Dataset,
        model: Model,
        device: str | torch.device,
    ) -> None:
        self.trainer_type = trainer_type
        self.params = params
        self.dataset = dataset
        self.model = model

        self.device = device

        # Setup
        self.batchprocessor = self._setup_batchprocessor()
        self.dataloader = self._setup_dataloader()
        self.optimizer = self._setup_optimizer()
        self.loss = self._setup_loss()
        self.regularizer = self._setup_regularizer()
        # TODO setup learning rate scheduler
        # TODO setup early stopping mechanism

    def train(self) -> float | None:
        """
        Core training loop.
        """
        self.model.to(self.device)

        logging.info(f"Start training: {self.params["epochs"]} epochs.")
        valid_loss = None
        for epoch in range(self.params["epochs"]):
            # Run training
            train_loss, train_regularization, train_total = self._train_one_epoch()

            # Run validation
            valid_loss = None
            # TODO validation!

            # Learning rate scheduler
            # TODO learning rate scheduler

            # Logging
            self._log_epoch_loss(epoch, train_loss, train_regularization, train_total, valid_loss)

            # Early stopping
            # TODO early stopping

        logging.info(f"Finished training.")

        score = valid_loss

        return score

    def _train_one_epoch(self) -> tuple[float, float, float]:
        """
        Runs a single training epoch.
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_regularization = 0.0
        epoch_total = 0.0

        for batch in self.dataloader:
            batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            inputs = self.batchprocessor.input_preparation(batch)
            predictions = self.model(inputs)

            # Target
            targets = self.batchprocessor.target_preparation(batch)

            # Criterion
            loss = self.loss(predictions, targets)

            # Regularization
            regularization = self.regularizer(self.model)

            # Gradient computation
            total = loss + regularization
            total.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_regularization += regularization.item()
            epoch_total += total.item()

        epoch_loss = epoch_loss / len(self.dataloader)
        epoch_regularization = epoch_regularization / len(self.dataloader)
        epoch_total = epoch_total / len(self.dataloader)

        return epoch_loss, epoch_regularization, epoch_total

    def _setup_batchprocessor(self) -> BatchProcessor:
        batchprocessor = BatchProcessorRegistry.get(self.dataset.dataset_type, self.model.model_type)()
        return batchprocessor

    def _setup_dataloader(self) -> Any:
        dataloader_cls = self.dataset.get_dataloader()

        dataloader = dataloader_cls(
            self.dataset,
            batch_size=self.params.get("batch_size", 2),
            shuffle=self.params.get("shuffle", True),
            collate_fn=self.dataset.collate_fn,
            drop_last=self.params.get("drop_last", False),
            num_workers=self.params.get("num_workers", 0),
        )

        return dataloader

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        optimizer_cls = OptimizerRegistry.get(self.params.get("optimizer", "adam"))
        optimizer = optimizer_cls(self.model.parameters(), self.params.get("learning_rate", 0.001))

        return optimizer

    def _setup_loss(self) -> Loss:
        loss_config = self.params["loss"]
        loss_type = loss_config["loss_type"]

        loss = LossRegistry.get(loss_type)(**loss_config)

        return loss

    def _setup_regularizer(self) -> Regularizer:
        regularizer_config = self.params.get("regularizer", None)
        if regularizer_config is None:
            return RegularizerRegistry.get("none")("none")

        regularizer_type = regularizer_config["regularizer_type"]
        regularizer = RegularizerRegistry.get(regularizer_type)(**regularizer_config)

        return regularizer

    def _log_epoch_loss(
        self,
        epoch: int,
        train_loss: float,
        train_regularization: float,
        train_total: float,
        valid_loss: float | None = None,
    ) -> None:
        # TODO better logging?
        if valid_loss is not None:
            logging.info(
                f"Epoch {epoch + 1:03d} | "
                f"Loss: {train_loss:.6f} | "
                f"Regularization: {train_regularization:.6f} | "
                f"Total: {train_total:.6f} | "
                f"Validation: {valid_loss:.6f}"
            )
        else:
            logging.info(
                f"Epoch {epoch + 1:03d} | "
                f"Loss: {train_loss:.6f} | "
                f"Regularization: {train_regularization:.6f} | "
                f"Total: {train_total:.6f} | "
            )

        # TODO mlflow|tensorboard logging?
