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
from typing import Any

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.registry import OptimizerRegistry

from ..base import Runner


class Trainer(Runner):
    def __init__(
        self,
        trainer_type: str,
        params: dict[str, Any],
        dataset: Dataset,
        model: Model,
        device: str | torch.device,
    ) -> None:
        super().__init__(params, dataset, model, device)

        self.trainer_type = trainer_type

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
            self._log_epoch_loss(train_loss, train_regularization, train_total, valid_loss, epoch)

            # Early stopping
            # TODO early stopping

        logging.info("Finished training.")

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

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        optimizer_cls = OptimizerRegistry.get(self.params.get("optimizer", "adam"))
        optimizer = optimizer_cls(self.model.parameters(), self.params.get("learning_rate", 0.001))

        return optimizer
