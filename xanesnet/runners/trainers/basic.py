# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Concrete ``BasicTrainer`` implementation for XANESNET."""

import logging

import torch

from xanesnet.checkpointing import Checkpointer
from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model
from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import tb_logger

from .base import Trainer
from .registry import TrainerRegistry


@TrainerRegistry.register("basic")
class BasicTrainer(Trainer):
    """Basic single-process trainer.

    Trains the model for a fixed number of epochs using the configuration
    supplied via the parent :class:`Trainer`.

    Args:
        dataset: Dataset to train on. Must provide ``train_subset``; ``valid_subset`` is optional.
        model: Model to train.
        device: Device identifier or :class:`torch.device` instance.
        checkpointer: Checkpoint manager for saving model states.
        encoding: Composed spectra encoding applied to targets before the loss is computed.
        batch_size: Number of samples per training batch.
        shuffle: Whether to shuffle training data each epoch.
        drop_last: Whether to drop the last incomplete training batch.
        num_workers: Number of data-loader worker processes.
        loss: Non-empty list of loss configurations.
        regularizer: Configuration for the regularizer.
        trainer_type: Identifier string for this trainer type.
        epochs: Total number of training epochs.
        learning_rate: Initial learning rate.
        optimizer: Optimizer name looked up via :class:`~xanesnet.components.OptimizerRegistry`.
        max_norm: Maximum gradient norm for clipping, or ``None`` to disable.
        lr_scheduler: Configuration for the per-epoch learning-rate scheduler.
        early_stopper: Configuration for the early-stopping criterion.
        validation_interval: Run validation every this many epochs when a validation subset is present.
        lr_warmup: Whether to apply a per-step linear warm-up phase.
        warmup_steps: Number of warm-up steps when ``lr_warmup=True``.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        device: str | torch.device,
        checkpointer: Checkpointer,
        encoding: SpectraEncoding,
        # runner params:
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        loss: list[Config],
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
        """Initialize ``BasicTrainer``."""
        super().__init__(
            dataset,
            model,
            device,
            checkpointer,
            encoding,
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
            max_norm,
            lr_scheduler,
            early_stopper,
            validation_interval,
            lr_warmup,
            warmup_steps,
        )
        self._global_step = 0
        self._warmup_complete = self.warmup_lr_scheduler is None

    def train(self) -> float | None:
        """Run the ordinary epoch loop with existing scheduling and selection.

        Returns:
            Best monitored score when restoring best weights, otherwise the
            final validation or training total. Returns ``None`` when no epoch
            completes.
        """
        self.early_stopper.start()
        self.model.to(self.device)
        self.loss.to(self.device)

        sample_batch = next(iter(self.dataloader))
        sample_batch.to(self.device)
        sample_inputs = self.batchprocessor.input_preparation(sample_batch)
        tb_logger.log_model_graph(self.model, sample_inputs)

        logging.info(f"Start training: {self.epochs} epochs.")
        train_total = None
        valid_total = None
        epoch = -1
        for epoch in range(self.epochs):
            train_loss, train_regularization, train_total = self._train_one_epoch()
            if self.valid_dataloader is not None and (
                epoch % self.validation_interval == 0 or epoch == self.epochs - 1
            ):
                valid_loss, valid_regularization, valid_total = self._validate_one_epoch()
            else:
                valid_loss, valid_regularization, valid_total = None, None, None

            self._log_epoch_loss(
                train_loss,
                train_regularization,
                train_total,
                epoch,
                valid_loss,
                valid_regularization,
                valid_total,
            )
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
            if self._warmup_complete:
                self.epoch_lr_scheduler.step()

            stopped = False
            if self.valid_dataloader is None:
                stopped = self.early_stopper.step(train_total, self.model, epoch)
            elif valid_total is not None:
                stopped = self.early_stopper.step(valid_total, self.model, epoch)

            if epoch == self.epochs - 1 or stopped:
                saved_checkpoint, checkpoint_name = self.checkpointer.save_checkpoint(epoch, self.model, self.optimizer)
            else:
                saved_checkpoint, checkpoint_name = self.checkpointer.step(epoch, self.model, self.optimizer)
            if saved_checkpoint:
                logging.info(f"Saved new checkpoint @ {self.checkpointer.save_dir}: {checkpoint_name}")
            if stopped:
                logging.info(f"EarlyStopper {self.early_stopper.early_stopper_type} fired in epoch {epoch}!")
                break

        logging.info("Finished training.")
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

    def _train_one_epoch(self) -> tuple[float, float, float]:
        """Run one training epoch.

        Returns:
            Tuple of ``(mean_loss, mean_regularization, mean_total)`` averaged
            over all target rows (observation-weighted).
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_regularization = 0.0
        epoch_total = 0.0
        epoch_rows = 0

        for batch in self.dataloader:
            batch.to(self.device)

            self.optimizer.zero_grad()

            inputs = self.batchprocessor.input_preparation(batch)
            elements = self.batchprocessor.element_preparation(batch)
            inputs = self.batchprocessor.encode_input(inputs, elements)
            targets = self.batchprocessor.target_preparation(batch)
            targets = self.batchprocessor.encode_target(targets, elements)

            predictions = self.model(**inputs)
            predictions = self.batchprocessor.prediction_preparation(batch, predictions)

            # Loss and regularization
            loss = self.loss(predictions, targets)
            regularization = self.regularizer(self.model)
            total = loss + regularization
            total.backward()
            if self.max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            # Per-step learning-rate warmup (no-op once warmup phase is over).
            self.step_warmup_scheduler()

            # Regularization is a batch-level scalar, but it is weighted by rows
            # here so the displayed total stays ``loss + regularization``.
            rows = int(targets.shape[0])
            epoch_loss += loss.item() * rows
            epoch_regularization += regularization.item() * rows
            epoch_total += total.item() * rows
            epoch_rows += rows

        epoch_loss = epoch_loss / epoch_rows
        epoch_regularization = epoch_regularization / epoch_rows
        epoch_total = epoch_total / epoch_rows

        return epoch_loss, epoch_regularization, epoch_total

    def _validate_one_epoch(self) -> tuple[float, float, float]:
        """Run one validation epoch (no gradient computation).

        Returns:
            Tuple of ``(mean_loss, mean_regularization, mean_total)`` averaged
            over all target rows (observation-weighted), matching the Lightning
            trainer's epoch metrics.
        """
        assert self.valid_dataloader is not None

        self.model.eval()

        valid_loss = 0.0
        valid_regularization = 0.0
        valid_total = 0.0
        valid_rows = 0

        with torch.no_grad():
            for batch in self.valid_dataloader:
                batch.to(self.device)

                inputs = self.batchprocessor.input_preparation(batch)
                elements = self.batchprocessor.element_preparation(batch)
                inputs = self.batchprocessor.encode_input(inputs, elements)
                targets = self.batchprocessor.target_preparation(batch)
                targets = self.batchprocessor.encode_target(targets, elements)

                predictions = self.model(**inputs)
                predictions = self.batchprocessor.prediction_preparation(batch, predictions)

                # Loss and regularization
                loss = self.loss(predictions, targets)
                regularization = self.regularizer(self.model)
                total = loss + regularization

                # Regularization is a batch-level scalar, but it is weighted by
                # rows here so the displayed total stays ``loss + regularization``.
                rows = int(targets.shape[0])
                valid_loss += loss.item() * rows
                valid_regularization += regularization.item() * rows
                valid_total += total.item() * rows
                valid_rows += rows

        valid_loss = valid_loss / valid_rows
        valid_regularization = valid_regularization / valid_rows
        valid_total = valid_total / valid_rows

        return valid_loss, valid_regularization, valid_total

    def step_warmup_scheduler(self) -> None:
        """Advance warm-up after an optimizer step until its configured end."""
        if self.warmup_lr_scheduler is None or self._warmup_complete:
            return
        self._global_step += 1
        self.warmup_lr_scheduler.step()
        if self._global_step >= self.warmup_steps:
            self._warmup_complete = True
