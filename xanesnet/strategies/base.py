# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Abstract base class for XANESNET training and inference strategies."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from xanesnet.checkpointing import Checkpointer
from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.serialization.config import Config


class Strategy(ABC):
    """Abstract base class for training and inference strategies.

    A strategy defines how one or more models are set up, trained, and used for
    inference (e.g. single model, ensemble, k-fold cross-validation). Subclasses
    must implement all abstract methods to provide concrete training and inference
    logic.

    Raises:
        ValueError: If neither ``trainer_config`` nor ``inferencer_config`` is
            provided at construction time.

    Args:
        strategy_type: Registry key identifying this strategy type.
        dataset: The dataset used for training or inference.
        model_config: Configuration for the model.
        weight_init: Weight initialization scheme name.
        weight_init_params: Additional parameters passed to the weight initializer.
        bias_init: Bias initialization scheme name.
        checkpoint_dir: Directory for saving checkpoint files, or
            ``None`` to disable checkpointing.
        checkpoint_interval: Number of epochs between checkpoints, or
            ``None`` to disable interval-based checkpointing.
        tensorboard_dir: Directory for TensorBoard event files, or
            ``None`` to disable TensorBoard logging.
        trainer_config: Configuration for the trainer. Either this or
            ``inferencer_config`` must be provided.
        inferencer_config: Configuration for the inferencer. Either this
            or ``trainer_config`` must be provided.

    Raises:
        ValueError: If both ``trainer_config`` and ``inferencer_config`` are ``None``.
    """

    def __init__(
        self,
        strategy_type: str,
        dataset: Dataset,
        model_config: Config,
        weight_init: str,
        weight_init_params: Config,
        bias_init: str,
        checkpoint_dir: str | Path | None,
        checkpoint_interval: int | None,
        tensorboard_dir: str | Path | None,
        trainer_config: Config | None = None,
        inferencer_config: Config | None = None,
    ) -> None:
        """Initialize shared strategy state."""
        self.strategy_type = strategy_type
        self.dataset = dataset
        self.model_config = model_config

        self.weight_init = weight_init
        self.weight_init_params = weight_init_params
        self.bias_init = bias_init
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.tensorboard_dir = tensorboard_dir
        self.trainer_config = trainer_config
        self.inferencer_config = inferencer_config

        if self.trainer_config is None and self.inferencer_config is None:
            raise ValueError("Either trainer_config or inferencer_config must be provided.")

        self.checkpointer: Checkpointer | None = None

    @abstractmethod
    def setup_models(self) -> None:
        """Instantiate the models used by this strategy.

        Must be called before ``init_model_weights``, ``setup_trainers``, or ``setup_inferencers``.
        """
        ...

    @abstractmethod
    def init_model_weights(self) -> None:
        """Apply weight and bias initialization to the strategy's models.

        Must be called after ``setup_models``.
        """
        ...

    @abstractmethod
    def set_state_dicts(self, state_dicts: list[dict]) -> None:
        """Load model state dicts from a list of previously saved dicts.

        Args:
            state_dicts: List of state dictionaries, one per model managed by this strategy.
        """
        ...

    @abstractmethod
    def setup_trainers(self, device: str | torch.device) -> None:
        """Instantiate the trainers used by this strategy.

        Must be called after ``setup_models`` and ``setup_checkpointer``.

        Args:
            device: The device on which training will be performed (e.g. ``"cpu"`` or ``"cuda"``).
        """
        ...

    @abstractmethod
    def run_training(self) -> list[Model]:
        """Execute the training loop and return the trained models.

        The base implementation logs the run start and returns an empty list.
        Subclasses must call ``super().run_training()`` and then return their
        trained models.

        Returns:
            List of trained ``Model`` instances.
        """
        logging.info("Start strategy...")
        return []

    @abstractmethod
    def setup_inferencers(self, device: str | torch.device) -> None:
        """Instantiate the inferencers used by this strategy.

        Must be called after ``setup_models``.

        Args:
            device: The device on which inference will be performed.
        """
        ...

    @abstractmethod
    def run_inference(self, predictions_save_path: str | Path | None) -> None:
        """Execute inference and optionally save predictions.

        The base implementation logs the run start. Subclasses must call
        ``super().run_inference(predictions_save_path)``.

        Args:
            predictions_save_path: Directory in which to write prediction
                output, or ``None`` to skip saving.
        """
        logging.info("Start strategy...")

    def setup_checkpointer(self) -> None:
        """Instantiate the ``Checkpointer`` for this strategy.

        Must be called after ``setup_models`` because the checkpointer is
        initialized with ``self.model_signature``.
        """
        self.checkpointer = Checkpointer(self.checkpoint_dir, self.checkpoint_interval, self.model_signature)

    @property
    @abstractmethod
    def model_signature(self) -> Config:
        """Return the model's configuration as a ``Config``.

        Returns:
            A ``Config`` representing the model's configuration signature.
        """
        ...

    @property
    @abstractmethod
    def signature(self) -> Config:
        """Return the full strategy configuration as a ``Config``.

        The base implementation returns a ``Config`` containing only the
        ``strategy_type`` key. Subclasses should call ``super().signature``
        and extend the result.

        Returns:
            A ``Config`` capturing the strategy configuration.
        """
        signature = Config(
            {
                "strategy_type": self.strategy_type,
            }
        )
        return signature
