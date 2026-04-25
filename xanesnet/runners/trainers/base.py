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
from abc import abstractmethod
from typing import Any

import torch

from xanesnet.checkpointing import Checkpointer
from xanesnet.components import LRSchedulerRegistry, OptimizerRegistry
from xanesnet.datasets import Dataset
from xanesnet.losses import Loss, LossRegistry
from xanesnet.models import Model
from xanesnet.regularizers import Regularizer, RegularizerRegistry
from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import tb_logger
from xanesnet.stoppers import EarlyStopper, EarlyStopperRegistry

from ..base import Runner


class Trainer(Runner):
    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        device: str | torch.device,
        checkpointer: Checkpointer,
        # runner params:
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        loss: Config,
        regularizer: Config,
        # trainer params:
        trainer_type: str,
        epochs: int,
        learning_rate: float,
        optimizer: str,
        max_norm: float | None,
        lr_scheduler: Config,
        early_stopper: Config,
        validation_interval: int,
        lr_warmup: bool,
        warmup_steps: int,
    ) -> None:
        super().__init__(dataset, model, device, batch_size, shuffle, drop_last, num_workers)

        self.checkpointer = checkpointer

        self.trainer_type = trainer_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.max_norm = max_norm
        self.lr_scheduler_config = lr_scheduler
        self.early_stopper_config = early_stopper
        self.validation_interval = validation_interval
        self.lr_warmup = lr_warmup
        self.warmup_steps = warmup_steps
        self.loss_config = loss
        self.regularizer_config = regularizer

        # Setup
        self.batchprocessor = self._setup_batchprocessor()
        self.dataloader = self._setup_train_dataloader()
        self.valid_dataloader = self._setup_valid_dataloader()
        self.optimizer = self._setup_optimizer()
        self.epoch_lr_scheduler = self._setup_epoch_lr_scheduler()  # per-epoch scheduler captures base lr
        self.warmup_lr_scheduler = self._setup_warmup_lr_scheduler()  # per-step warmup scheduler
        self._global_step = 0
        self._warmup_complete = self.warmup_lr_scheduler is None
        self.early_stopper = self._setup_early_stopper()
        self.loss = self._setup_loss()
        self.regularizer = self._setup_regularizer()

    def _setup_loss(self) -> Loss:
        loss_config = self.loss_config
        if loss_config is None:
            raise ValueError("Loss config is required but was not provided.")
        loss_type = loss_config.get_str("loss_type")

        loss = LossRegistry.get(loss_type)(**loss_config.as_kwargs())

        return loss

    def _setup_regularizer(self) -> Regularizer:
        regularizer_config = self.regularizer_config
        if regularizer_config is None:
            raise ValueError("Regularizer config is required but was not provided.")
        regularizer_type = regularizer_config.get_str("regularizer_type")

        regularizer = RegularizerRegistry.get(regularizer_type)(**regularizer_config.as_kwargs())

        return regularizer

    def train(self) -> float | None:
        """
        Core training loop.
        """
        self.model.to(self.device)

        # Log model graph (once, using first training batch as example input)
        sample_batch = next(iter(self.dataloader))
        sample_batch.to(self.device)
        sample_inputs = self.batchprocessor.input_preparation(sample_batch)
        tb_logger.log_model_graph(self.model, sample_inputs)

        logging.info(f"Start training: {self.epochs} epochs.")

        train_total = None
        valid_total = None

        epoch = -1
        for epoch in range(self.epochs):
            # Run training
            train_loss, train_regularization, train_total = self._train_one_epoch()

            # Run validation on interval or last epoch
            if self.valid_dataloader is not None and (
                epoch % self.validation_interval == 0 or epoch == self.epochs - 1
            ):
                valid_loss, valid_regularization, valid_total = self._validate_one_epoch()
            else:
                valid_loss, valid_regularization, valid_total = None, None, None

            # Logging
            self._log_epoch_loss(
                train_loss,
                train_regularization,
                train_total,
                valid_loss,
                valid_regularization,
                valid_total,
                epoch,
            )

            # TensorBoard logging
            tb_logger.log_epoch_metrics(
                epoch,
                train_loss,
                train_regularization,
                train_total,
                valid_loss,
                valid_regularization,
                valid_total,
            )
            tb_logger.log_learning_rate(epoch, self.optimizer.param_groups[0]["lr"])
            tb_logger.log_model_weights(epoch, self.model)

            # Per-epoch learning-rate scheduler.
            if self._warmup_complete:
                self.epoch_lr_scheduler.step()

            # Early stopping check
            stopped = False
            if self.valid_dataloader is None:
                stopped = self.early_stopper.step(train_total, self.model, epoch)
            elif valid_total is not None:
                stopped = self.early_stopper.step(valid_total, self.model, epoch)

            # Checkpointing
            if epoch == self.epochs - 1 or stopped:
                saved_checkpoint, checkpoint_name = self.checkpointer.save_checkpoint(epoch, self.model, self.optimizer)
            else:
                saved_checkpoint, checkpoint_name = self.checkpointer.step(epoch, self.model, self.optimizer)
            if saved_checkpoint:
                logging.info(f"Saved new checkpoint @ {self.checkpointer.save_dir}: {checkpoint_name}")

            # Early stopping trigger
            if stopped:
                logging.info(f"EarlyStopper {self.early_stopper.early_stopper_type} fired in epoch {epoch}!")
                break

        logging.info("Finished training.")

        # Restore best model / Taking last model
        if self.early_stopper.restore_best:
            score, best_epoch = self.early_stopper.restore(self.model)
            if score is not None and best_epoch is not None:
                logging.info(f"Restored best model from epoch {best_epoch} with score {score}.")
            else:
                logging.warning(
                    f"Did not find a best model."
                    f" Something might be wrong in your EarlyStopper {self.early_stopper.early_stopper_type}."
                )
        else:
            if valid_total is not None:
                score = valid_total
                logging.info(f"Using last model from epoch {epoch} as final model with validation score {score}.")
            else:
                score = train_total
                logging.info(f"Using last model from epoch {epoch} as final model with training score {score}.")

        return score

    @abstractmethod
    def _train_one_epoch(self) -> tuple[float, float, float]:
        """
        Runs a single training epoch.
        """
        ...

    @abstractmethod
    def _validate_one_epoch(self) -> tuple[float, float, float]:
        """
        Runs a single validation epoch.
        """
        ...

    def _setup_train_dataloader(self) -> Any:
        if self.dataset.train_subset is None:
            raise ValueError("Training subset is required but was not provided.")

        dataloader_cls = self.dataset.get_dataloader()

        dataloader = dataloader_cls(
            self.dataset.train_subset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.dataset.collate_fn,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            prefetch_factor=None if self.num_workers == 0 else 2,
        )

        return dataloader

    def _setup_valid_dataloader(self) -> Any | None:
        if self.dataset.valid_subset is None:
            raise ValueError("Validation subset is required but was not provided.")

        dataloader_cls = self.dataset.get_dataloader()

        dataloader = dataloader_cls(
            self.dataset.valid_subset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            collate_fn=self.dataset.collate_fn,
            drop_last=False,  # Keep all validation samples
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False if self.num_workers == 0 else True,
            prefetch_factor=None if self.num_workers == 0 else 2,
        )

        return dataloader

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        optimizer_cls = OptimizerRegistry.get(self.optimizer_type)
        optimizer = optimizer_cls(self.model.parameters(), lr=self.learning_rate)  # type: ignore

        return optimizer

    def _setup_epoch_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Build the main per-epoch learning-rate scheduler.
        """
        lr_scheduler_type = self.lr_scheduler_config.get_str("lr_scheduler_type")

        # We have to remove 'lr_scheduler_type'!
        lr_scheduler_kwargs = self.lr_scheduler_config.as_kwargs()
        key_to_remove = "lr_scheduler_type"
        config_wo_type = {k: v for k, v in lr_scheduler_kwargs.items() if k != key_to_remove}

        lr_scheduler_cls = LRSchedulerRegistry.get(lr_scheduler_type)
        lr_scheduler = lr_scheduler_cls(self.optimizer, **config_wo_type)

        return lr_scheduler

    def _setup_warmup_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """
        Build the per-step warmup scheduler (or ``None`` if disabled).
        """
        if not self.lr_warmup or self.warmup_steps <= 0:
            return None

        return torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-7,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

    def step_warmup_scheduler(self) -> None:
        """
        Advance the per-step warmup scheduler by one optimizer step.

        Should be called by subclasses from inside the training batch loop, after ``optimizer.step()``.
        """
        if self.warmup_lr_scheduler is None or self._warmup_complete:
            return

        self._global_step += 1
        self.warmup_lr_scheduler.step()
        if self._global_step >= self.warmup_steps:
            self._warmup_complete = True

    def _setup_early_stopper(self) -> EarlyStopper:
        early_stopper_type = self.early_stopper_config.get_str("early_stopper_type")

        early_stopper_cls = EarlyStopperRegistry.get(early_stopper_type)
        early_stopper = early_stopper_cls(**self.early_stopper_config.as_kwargs())

        return early_stopper
