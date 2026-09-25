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
#   ...

"""Bootstrap ensemble strategy for XANESNET."""

import copy
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model, ModelRegistry
from xanesnet.runners.inferencers import Inferencer, InferencerRegistry
from xanesnet.runners.trainers import Trainer, TrainerRegistry
from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import tb_logger

from .base import Strategy
from .registry import StrategyRegistry


@StrategyRegistry.register("bootstrap")
class Bootstrap(Strategy):
    """Sequential bootstrap-ensemble training and aggregate inference strategy.

    The strategy owns ``n_models`` independent model instances with identical
    architecture. During training, each member is trained on a bootstrap
    resample of the training subset (sampling with replacement). During
    inference, all member models are evaluated on the same batches and their
    predictions are reduced to mean and energy/channel-wise standard deviation
    by the ensemble inferencer.

    Args:
        strategy_type: Registry key identifying this strategy type.
        dataset: Dataset used for training or inference.
        model_config: Configuration for the model.
        encoding: Composed spectra encoding forwarded to the trainers and inferencer.
        weight_init: Weight initialization scheme name.
        weight_init_params: Additional weight-initializer parameters.
        bias_init: Bias initialization scheme name.
        checkpoint_dir: Directory for checkpoints, or ``None``.
        checkpoint_interval: Epoch interval between checkpoints, or ``None``.
        tensorboard_dir: Directory for TensorBoard event files, or ``None``.
        trainer_config: Trainer configuration for training mode.
        inferencer_config: Inferencer configuration for inference mode.
        n_models: Number of bootstrap ensemble members.
        sample_fraction: Fraction of the training subset drawn for each
            bootstrap resample. The effective sample size is at least one.
    """

    def __init__(
        self,
        strategy_type: str,
        dataset: Dataset,
        model_config: Config,
        encoding: SpectraEncoding,
        weight_init: str,
        weight_init_params: Config,
        bias_init: str,
        checkpoint_dir: str | Path | None,
        checkpoint_interval: int | None,
        tensorboard_dir: str | Path | None,
        trainer_config: Config | None,
        inferencer_config: Config | None,
        # bootstrap arguments:
        n_models: int,
        sample_fraction: float,
    ) -> None:
        """Initialize the bootstrap ensemble strategy."""
        super().__init__(
            strategy_type,
            dataset,
            model_config,
            encoding,
            weight_init,
            weight_init_params,
            bias_init,
            checkpoint_dir,
            checkpoint_interval,
            tensorboard_dir,
            trainer_config,
            inferencer_config,
        )

        self.n_models = n_models
        self.sample_fraction = sample_fraction

        self.models: list[Model] = []
        self.trainers: list[Trainer | None] = []
        self.inferencer: Inferencer | None = None

    def _bootstrap_dataset(self) -> Dataset:
        """Return a dataset copy with a bootstrap-resampled training subset.

        Returns:
            A shallow copy of ``self.dataset`` whose training subset contains
            a bootstrap resample of the original training indices.
        """
        train_indices = self.dataset.get_subset_indices(0)
        if train_indices is None:
            train_indices = list(range(len(self.dataset)))

        n_samples = len(train_indices)
        sample_size = max(1, int(n_samples * self.sample_fraction))
        bootstrap_indices = np.random.choice(train_indices, size=sample_size, replace=True).tolist()

        dataset_boot = copy.copy(self.dataset)
        subsets: list[Subset] = [Subset(self.dataset, bootstrap_indices)]
        valid_subset = self.dataset.valid_subset
        if valid_subset is not None:
            subsets.append(valid_subset)
        dataset_boot._subsets = subsets

        return dataset_boot

    def _dataset_for_model(self) -> Dataset:
        """Return a bootstrap-resampled dataset view for one model.

        Returns:
            A shallow dataset copy whose training subset is bootstrap-resampled.
        """
        return self._bootstrap_dataset()

    def setup_models(self) -> None:
        """Instantiate ``n_models`` independent model copies from ``model_config``."""
        model_type = self.model_config.get_str("model_type")
        model_cls = ModelRegistry.get(model_type)

        self.models = []
        for model_idx in range(self.n_models):
            logging.info(f"Initializing bootstrap model {model_idx + 1}/{self.n_models}: {model_type}")
            self.models.append(model_cls(**self.model_config.as_kwargs()))

    def init_model_weights(self) -> None:
        """Apply configured weight and bias initialization to every model.

        Members are initialized independently through the globally seeded
        PyTorch random-number generator.

        Raises:
            ValueError: If ``setup_models`` has not been called.
        """
        if len(self.models) == 0:
            raise ValueError("Cannot initialize model weights because models are not initialized.")

        logging.info(f"Initializing weights with '{self.weight_init}' and bias with '{self.bias_init}'")
        for model_idx, model in enumerate(self.models):
            logging.info(f"Initializing bootstrap model {model_idx + 1}/{self.n_models} weights.")
            model.init_weights(self.weight_init, self.bias_init, **self.weight_init_params.as_kwargs())

    def set_state_dicts(self, state_dicts: list[Mapping[str, Any]]) -> None:
        """Load one state dictionary into each bootstrap member.

        Args:
            state_dicts: State dictionaries to load, one per model.

        Raises:
            ValueError: If models are not initialized or the number of state
                dictionaries does not match the number of models.
        """
        if len(self.models) == 0:
            raise ValueError("Cannot load state dicts because models are not initialized.")
        if len(state_dicts) != len(self.models):
            raise ValueError(f"Expected {len(self.models)} state dicts, got {len(state_dicts)}.")

        for model, state_dict in zip(self.models, state_dicts, strict=True):
            model.load_state_dict(state_dict)

    def setup_trainers(self, device: str | torch.device) -> None:
        """Instantiate one trainer per bootstrap member.

        Each trainer receives a dataset copy whose training subset is a
        bootstrap resample of the original training data.

        Must be called after ``setup_models`` and ``setup_checkpointer``.

        Args:
            device: The device on which training will be performed.

        Raises:
            ValueError: If models, trainer config, or checkpointer are not initialized.
        """
        if len(self.models) == 0:
            raise ValueError("Cannot setup trainers because models are not initialized.")
        if self.trainer_config is None:
            raise ValueError("Can not setup trainers because there is no trainer config.")
        if self.checkpointer is None:
            raise ValueError("Can not setup trainers because checkpointer is not instantiated.")

        trainer_type = self.trainer_config.get_str("trainer_type")
        trainer_cls = TrainerRegistry.get(trainer_type)

        self.trainers = []
        for model_idx, model in enumerate(self.models):
            logging.info(f"Initializing trainer {model_idx + 1}/{self.n_models}: {trainer_type}")
            dataset_model = self._dataset_for_model()
            trainer = trainer_cls(
                **self.trainer_config.as_kwargs(),
                dataset=dataset_model,
                model=model,
                device=device,
                checkpointer=self.checkpointer,
                encoding=self.encoding,
            )
            self.trainers.append(trainer)

    def run_training(self) -> list[Model]:
        """Train all bootstrap members sequentially and return them.

        Must be called after ``setup_trainers``.

        Returns:
            List of trained bootstrap member models. Trainer instances are
            released after their corresponding member finishes to avoid keeping
            optimizer state alive during later member training.

        Raises:
            ValueError: If trainers or models are not initialized.
        """
        if len(self.models) == 0:
            raise ValueError("Cannot run training because models are not initialized.")
        if len(self.trainers) != len(self.models):
            raise ValueError("Cannot run training because trainers are not initialized for every model.")

        super().run_training()

        assert self.checkpointer is not None

        for model_idx, trainer in enumerate(self.trainers):
            if trainer is None:
                raise ValueError("Cannot run training because trainers are not initialized for every model.")

            logging.info(f"Training bootstrap model {model_idx + 1}/{self.n_models}.")
            self.checkpointer.new_model()

            try:
                if self.tensorboard_dir is not None:
                    tb_logger.new_run(Path(self.tensorboard_dir) / f"model_{model_idx}")

                trainer.train()
            finally:
                tb_logger.close()
                self.models[model_idx].to(torch.device("cpu"))
                self.trainers[model_idx] = None

        return self.models

    def setup_inferencers(self, device: str | torch.device) -> None:
        """Instantiate the ensemble inferencer for all loaded models.

        Must be called after ``setup_models`` and ``set_state_dicts``.

        Args:
            device: The device on which inference will be performed.

        Raises:
            ValueError: If models or inferencer config are not initialized.
        """
        if len(self.models) == 0:
            raise ValueError("Can not setup inferencers because models are not initialized.")
        if self.inferencer_config is None:
            raise ValueError("Can not setup inferencers because there is no inferencer config.")

        logging.info("Initializing inferencer: ensemble")

        inferencer_kwargs = self.inferencer_config.as_kwargs()
        inferencer_kwargs["inferencer_type"] = "ensemble"
        inferencer = InferencerRegistry.create(
            "ensemble",
            **inferencer_kwargs,
            dataset=self.dataset,
            models=self.models,
            device=device,
            encoding=self.encoding,
        )

        self.inferencer = inferencer

    def run_inference(self, predictions_save_path: str | Path | None) -> None:
        """Run aggregate ensemble inference and optionally save predictions.

        Args:
            predictions_save_path: Directory in which to write prediction
                output, or ``None`` to skip saving.

        Raises:
            ValueError: If ``setup_inferencers`` has not been called.
        """
        if self.inferencer is None:
            raise ValueError("Cannot run inference because the Inferencer is not initialized.")

        super().run_inference(predictions_save_path)

        self.inferencer.infer(predictions_save_path)

    @property
    def model_signature(self) -> Config:
        """Return the shared model architecture signature.

        Returns:
            A ``Config`` representing the bootstrap members' model signature.

        Raises:
            ValueError: If ``setup_models`` has not been called.
        """
        if len(self.models) == 0:
            raise ValueError("Models are not initialized. Cannot retrieve signature.")

        return self.models[0].signature

    @property
    def signature(self) -> Config:
        """Return the strategy configuration as a ``Config``.

        Returns:
            A ``Config`` capturing the strategy configuration.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "n_models": self.n_models,
                "sample_fraction": self.sample_fraction,
            }
        )
        return signature
