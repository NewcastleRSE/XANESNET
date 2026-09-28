# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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

import torch

from xanesnet.checkpointing import Checkpointer
from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model
from xanesnet.serialization.config import Config

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
        encoding: Composed spectra encoding applied to targets before the loss
            is computed.
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

    def _train_one_epoch(self) -> tuple[float, float, float]:
        """Run one training epoch.

        Returns:
            Tuple of ``(mean_loss, mean_regularization, mean_total)`` averaged
            over all batches in the training data loader.
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_regularization = 0.0
        epoch_total = 0.0

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

            epoch_loss += loss.item()
            epoch_regularization += regularization.item()
            epoch_total += total.item()

        epoch_loss = epoch_loss / len(self.dataloader)
        epoch_regularization = epoch_regularization / len(self.dataloader)
        epoch_total = epoch_total / len(self.dataloader)

        return epoch_loss, epoch_regularization, epoch_total

    def _validate_one_epoch(self) -> tuple[float, float, float]:
        """Run one validation epoch (no gradient computation).

        Returns:
            Tuple of ``(mean_loss, mean_regularization, mean_total)`` averaged
            over all batches in the validation data loader.
        """
        assert self.valid_dataloader is not None

        self.model.eval()

        valid_loss = 0.0
        valid_regularization = 0.0
        valid_total = 0.0

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

                valid_loss += loss.item()
                valid_regularization += regularization.item()
                valid_total += total.item()

        valid_loss = valid_loss / len(self.valid_dataloader)
        valid_regularization = valid_regularization / len(self.valid_dataloader)
        valid_total = valid_total / len(self.valid_dataloader)

        return valid_loss, valid_regularization, valid_total
