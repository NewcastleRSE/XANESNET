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

"""Abstract base class for XANESNET training and inference strategies."""

import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from xanesnet.checkpointing import Checkpointer
from xanesnet.datasets import Dataset
from xanesnet.encodings import CombinedEncoding, SpectraEncoding
from xanesnet.models import Model
from xanesnet.runners.inferencers.base import Inferencer
from xanesnet.runners.trainers.base import Trainer
from xanesnet.serialization.auto_config import resolve_auto_encoding_config
from xanesnet.serialization.config import Config


class Strategy(ABC):
    """Abstract base class for training and inference strategies.

    A strategy defines how one or more models are set up, trained, and used for
    inference (e.g. single model, deep ensemble, snapshot ensemble, bootstrap).
    Subclasses must implement all abstract methods to provide concrete training
    and inference logic.

    Args:
        strategy_type: Registry key identifying this strategy type.
        dataset: The dataset used for training or inference.
        weight_init: Weight initialization scheme name.
        weight_init_params: Additional parameters passed to the weight initializer.
        bias_init: Bias initialization scheme name.
        checkpoint_dir: Directory for saving checkpoint files, or ``None`` to disable checkpointing.
        checkpoint_interval: Number of epochs between checkpoints, or ``None`` to disable interval-based checkpointing.
        tensorboard_dir: Directory for TensorBoard event files, or ``None`` to disable TensorBoard logging.
        trainer_config: Configuration for the trainer. Either this or ``inferencer_config`` must be provided.
        inferencer_config: Configuration for the inferencer. Either this or ``trainer_config`` must be provided.

    Raises:
        ValueError: If both ``trainer_config`` and ``inferencer_config`` are ``None``.
    """

    def __init__(
        self,
        strategy_type: str,
        dataset: Dataset,
        weight_init: str,
        weight_init_params: Config,
        bias_init: str,
        checkpoint_dir: str | Path | None,
        checkpoint_interval: int | None,
        tensorboard_dir: str | Path | None,
        trainer_config: Config | None,
        inferencer_config: Config | None,
    ) -> None:
        """Initialize shared strategy state."""
        self.strategy_type = strategy_type
        self.dataset = dataset
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

        self.n_models = 1
        self.model_splits: list[list[list[int]]] | None = None
        self.model_datasets: list[Dataset] = []
        self.encoding: SpectraEncoding | None = None
        self.models: list[Model] = []
        self.checkpointer: Checkpointer | None = None
        self.trainers: list[Trainer | None] = []
        self.inferencer: Inferencer | None = None

    @abstractmethod
    def setup_models(self, model_config: Config) -> None:
        """Instantiate the models from the resolved model configuration.

        Must be called before ``init_model_weights``, ``setup_trainers``, or ``setup_inferencers``.

        Args:
            model_config: Fully resolved configuration for the models managed by this strategy.
        """
        ...

    def init_model_weights(self) -> None:
        """Apply weight and bias initialization to the strategy's models.

        Must be called after ``setup_models``.

        Raises:
            ValueError: If no models have been initialized.
        """
        if not self.models:
            raise ValueError("Cannot initialize model weights because models are not initialized.")

        logging.info(f"Initializing weights with '{self.weight_init}' and bias with '{self.bias_init}'")
        for model in self.models:
            model.init_weights(self.weight_init, self.bias_init, **self.weight_init_params.as_kwargs())

    def set_state_dicts(self, state_dicts: list[Mapping[str, Any]]) -> None:
        """Load one state dictionary into each model.

        Args:
            state_dicts: Model state dictionaries in strategy model order.

        Raises:
            ValueError: If models are uninitialized or state counts mismatch.
        """
        if not self.models:
            raise ValueError("Cannot load state dicts because models are not initialized.")
        if len(state_dicts) != len(self.models):
            raise ValueError(f"Expected {len(self.models)} state dicts, got {len(state_dicts)}.")
        for model, state_dict in zip(self.models, state_dicts, strict=True):
            model.load_state_dict(state_dict)

    def setup_model_splits(self, model_splits: list[list[list[int]]] | None = None) -> list[list[list[int]]]:
        """Draw or restore dataset splits specific to individual models.

        Must be called before ``setup_encoding`` or ``setup_trainers``.
        The default implementation applies the dataset's configured splits and
        gives every model the same plan.

        Args:
            model_splits: Previously drawn subset indices, one ordered list of
                split index lists per model, or ``None`` to draw new splits.

        Raises:
            ValueError: If restored plans do not contain one entry per model.
        """
        if model_splits is None:
            if not self.dataset.subsets:
                self.dataset.setup_splits()
            configured_splits = self.dataset.get_all_subset_indices()
            if not configured_splits:
                configured_splits = [list(range(len(self.dataset)))]
            model_splits = [[list(indices) for indices in configured_splits] for _ in range(self.n_models)]
        if len(model_splits) != self.n_models:
            raise ValueError(f"Expected {self.n_models} model splits, got {len(model_splits)}.")
        self.model_splits = model_splits
        self.model_datasets = []
        for splits in model_splits:
            model_dataset = copy.copy(self.dataset)
            model_dataset._subsets = [Subset(self.dataset, indices) for indices in splits]
            self.model_datasets.append(model_dataset)
        return model_splits

    def setup_encoding(self, config: Config) -> SpectraEncoding:
        """Resolve and build the strategy's shared encoding.

        Args:
            config: Configuration whose encoding fields may require automatic
                resolution. Inference configs already contain concrete values.

        Raises:
            ValueError: If a training strategy has no model splits, or if the
                prepared dataset sample does not provide a spectral energy grid.

        Returns:
            The initialized shared encoding.
        """
        if self.trainer_config is not None and self.model_splits is None:
            raise ValueError("Cannot setup encoding before model splits are initialized.")

        resolved_config = resolve_auto_encoding_config(config, self._dataset_for_encoding())
        encoding = CombinedEncoding.from_configs(resolved_config.get_config_list("encodings"))
        energies = getattr(self.dataset[0], "energies", None)
        if energies is None or energies.ndim == 0:
            raise ValueError("Prepared dataset sample must provide a spectral energy grid.")
        encoding.prepare(int(energies.shape[-1]))
        self.encoding = encoding
        return encoding

    @abstractmethod
    def setup_trainers(self, device: str | torch.device) -> None:
        """Instantiate the trainers used by this strategy.

        Must be called after ``setup_models``, ``setup_checkpointer``,
        ``setup_model_splits``, and ``setup_encoding``.

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

        Must be called after ``setup_models`` and ``setup_encoding``.

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

    def _dataset_for_encoding(self) -> Dataset:
        """Return the data view used to fit this strategy's encoding."""
        return self.dataset

    @property
    def model_signature(self) -> Config:
        """Return the first model's configuration as the strategy signature.

        Returns:
            A ``Config`` representing the model's configuration signature.

        Raises:
            ValueError: If models have not been initialized.
        """
        if not self.models:
            raise ValueError("Models are not initialized. Cannot retrieve signature.")

        return self.models[0].signature

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
