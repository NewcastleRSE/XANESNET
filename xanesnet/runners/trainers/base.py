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

from xanesnet.components import LRSchedulerRegistry, OptimizerRegistry
from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.stoppers import EarlyStopper, EarlyStopperRegistry

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
        self.lr_scheduler = self._setup_lr_scheduler()
        self.early_stopper = self._setup_early_stopper()
        self.loss = self._setup_loss()
        self.regularizer = self._setup_regularizer()

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
            self.lr_scheduler.step()

            # Logging
            self._log_epoch_loss(train_loss, train_regularization, train_total, valid_loss, epoch)

            # Early stopping
            if self.early_stopper.step(valid_loss, self.model, epoch):
                logging.info(f"EarlyStopper {self.early_stopper.stopper_type} fired in epoch {epoch}!")
                break

        logging.info("Finished training.")

        if self.early_stopper.restore_best:
            score, best_epoch = self.early_stopper.restore(self.model)
            if score is not None and best_epoch is not None:
                logging.info(f"Restored best model from epoch {best_epoch} with score {score}.")
            else:
                logging.warning(
                    f"Did not find a best model."
                    f" Something might be wrong in your EarlyStopper {self.early_stopper.stopper_type}."
                )
        else:
            score = valid_loss
            logging.info(f"Using last model as final model.")

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
        optimizer_cls = OptimizerRegistry.get(self.params["optimizer"])
        optimizer = optimizer_cls(self.model.parameters(), self.params["learning_rate"])

        return optimizer

    def _setup_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        lr_scheduler_config = self.params["lr_scheduler"]
        lr_scheduler_type = lr_scheduler_config["lr_scheduler_type"]
        lr_scheduler_params = lr_scheduler_config["params"]

        lr_scheduler_cls = LRSchedulerRegistry.get(lr_scheduler_type)
        lr_scheduler = lr_scheduler_cls(self.optimizer, **lr_scheduler_params)

        return lr_scheduler

    def _setup_early_stopper(self) -> EarlyStopper:
        early_stopper_config = self.params["early_stopper"]
        early_stopper_type = early_stopper_config["early_stopper_type"]
        early_stopper_params = early_stopper_config["params"]

        early_stopper_cls = EarlyStopperRegistry.get(early_stopper_type)
        early_stopper = early_stopper_cls(early_stopper_type, **early_stopper_params)

        return early_stopper
