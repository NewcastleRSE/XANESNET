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

"""Abstract base class for all XANESNET trainers."""

from abc import abstractmethod
from typing import Any

import torch

from xanesnet.checkpointing import Checkpointer
from xanesnet.components import LRSchedulerRegistry, OptimizerRegistry
from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.losses import CombinedLoss, Loss, LossRegistry
from xanesnet.models import Model
from xanesnet.regularizers import Regularizer, RegularizerRegistry
from xanesnet.serialization.config import Config
from xanesnet.stoppers import EarlyStopper, EarlyStopperRegistry

from ..base import Runner


class Trainer(Runner):
    """Abstract base class for all XANESNET trainers.

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
        trainer_type: Identifier string for the concrete trainer type.
        epochs: Total number of training epochs.
        learning_rate: Initial learning rate.
        optimizer: Optimizer name (looked up via :class:`OptimizerRegistry`).
        max_norm: Maximum gradient norm for clipping, or ``None`` to disable.
        lr_scheduler: Configuration for the per-epoch learning-rate scheduler.
        early_stopper: Configuration for the early-stopping criterion.
        validation_interval: Run validation every this many epochs when a
            validation subset is present.
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
        """Initialize ``Trainer``."""
        super().__init__(dataset, model, device, encoding, batch_size, shuffle, drop_last, num_workers)

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
        self.loss = self._setup_loss()
        self.regularizer = self._setup_regularizer()
        self.early_stopper = self._setup_early_stopper()

    @abstractmethod
    def train(self) -> float | None:
        """Execute the concrete backend's training lifecycle.

        Returns:
            Final score used for model selection: the best validation total
            (when ``early_stopper.restore_best=True``), the last validation
            total (when a validation set is present), or the last training
            total otherwise. Returns ``None`` if training did not complete a
            single epoch.
        """
        ...

    def _setup_loss(self) -> Loss:
        """Instantiate a :class:`~xanesnet.losses.CombinedLoss` from the list of loss
        configurations.

        Each entry in the configuration list must provide ``loss_type`` and may
        provide ``loss_weight`` (a positive number).  Missing weights default to
        ``1.0``; all weights are normalized to sum to one before constructing
        :class:`~xanesnet.losses.CombinedLoss`.

        Returns:
            A configured :class:`~xanesnet.losses.CombinedLoss` instance.

        Raises:
            ValueError: If the loss configuration list is empty or not provided.
        """
        loss_configs = self.loss_config
        if not loss_configs:
            raise ValueError("Loss config list is required but was not provided.")

        losses: list[Loss] = []
        raw_weights: list[float] = []

        for item_config in loss_configs:
            loss_type = item_config.get_str("loss_type")
            raw_weight = item_config.get_optional_float("loss_weight")
            raw_weights.append(raw_weight if raw_weight is not None else 1.0)
            kwargs = {k: v for k, v in item_config.as_kwargs().items() if k != "loss_weight"}
            losses.append(LossRegistry.create(loss_type, **kwargs))

        return CombinedLoss(losses, raw_weights)

    def _setup_regularizer(self) -> Regularizer:
        """Instantiate the regularizer from configuration.

        Returns:
            A configured :class:`Regularizer` instance.

        Raises:
            ValueError: If the regularizer configuration is not provided.
        """
        regularizer_config = self.regularizer_config
        if regularizer_config is None:
            raise ValueError("Regularizer config is required but was not provided.")
        regularizer_type = regularizer_config.get_str("regularizer_type")

        regularizer = RegularizerRegistry.create(regularizer_type, **regularizer_config.as_kwargs())

        return regularizer

    def _setup_train_dataloader(self) -> Any:
        """Build a data loader over the training subset.

        Returns:
            A configured data-loader instance.

        Raises:
            ValueError: If the dataset provides no training subset.
        """
        if self.dataset.train_subset is None:
            raise ValueError("Training subset is required but was not provided.")

        dataloader = self._build_dataloader(
            self.dataset.train_subset,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

        return dataloader

    def _setup_valid_dataloader(self) -> Any | None:
        """Build a data loader over the validation subset.

        Returns:
            A configured data-loader instance, or ``None`` when the dataset
            provides no non-empty validation subset.
        """
        if self.dataset.valid_subset is None or len(self.dataset.valid_subset) == 0:
            return None

        dataloader = self._build_dataloader(
            self.dataset.valid_subset,
            shuffle=False,  # No need to shuffle validation data
            drop_last=False,  # Keep all validation samples
        )

        return dataloader

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Instantiate the optimizer for the model parameters.

        Returns:
            A configured :class:`torch.optim.Optimizer` instance.
        """
        optimizer_cls = OptimizerRegistry.get(self.optimizer_type)
        optimizer = optimizer_cls(self.model.parameters(), lr=self.learning_rate)  # type: ignore

        return optimizer

    def _setup_epoch_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Build the main per-epoch learning-rate scheduler.

        Returns:
            A configured :class:`torch.optim.lr_scheduler.LRScheduler` instance.
        """
        lr_scheduler_type = self.lr_scheduler_config.get_str("lr_scheduler_type")

        # Remove 'lr_scheduler_type' before passing kwargs to the scheduler constructor.
        lr_scheduler_kwargs = self.lr_scheduler_config.as_kwargs()
        config_wo_type = {k: v for k, v in lr_scheduler_kwargs.items() if k != "lr_scheduler_type"}

        lr_scheduler_cls = LRSchedulerRegistry.get(lr_scheduler_type)
        lr_scheduler = lr_scheduler_cls(self.optimizer, **config_wo_type)

        return lr_scheduler

    def _setup_warmup_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Build the per-step linear warm-up scheduler.

        Returns:
            A :class:`torch.optim.lr_scheduler.LinearLR` instance when
            ``lr_warmup=True`` and ``warmup_steps > 0``, otherwise ``None``.
        """
        if not self.lr_warmup or self.warmup_steps <= 0:
            return None

        return torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-7,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

    def _setup_early_stopper(self) -> EarlyStopper:
        """Construct the configured stopper for the concrete backend to drive.

        Returns:
            Registered stopping implementation, including best-weight tracking
            when configured, without starting a training loop.
        """
        stopper_type = self.early_stopper_config.get_str("early_stopper_type")
        return EarlyStopperRegistry.create(stopper_type, **self.early_stopper_config.as_kwargs())
