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

"""K-fold cross-validation strategy for XANESNET."""

import copy
import logging
from collections.abc import Iterator, Mapping
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


@StrategyRegistry.register("kfold")
class KFold(Strategy):
    """Repeated k-fold cross-validation ensemble training strategy.

    The strategy trains one model per fold on a shuffled partition of the full
    dataset. Each fold uses the holdout partition as validation during training.
    There are ``n_splits * n_repeats`` fold models in total. All fold models are
    retained and their predictions are aggregated by the ensemble inferencer.

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
        n_splits: Number of folds per repeat.
        n_repeats: Number of times to repeat the k-fold split.
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
        # k-fold arguments:
        n_splits: int,
        n_repeats: int,
    ) -> None:
        """Initialize the k-fold cross-validation strategy."""
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

        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_models = n_splits * n_repeats

        self.models: list[Model] = []
        self.trainers: list[Trainer | None] = []
        self.inferencer: Inferencer | None = None
        self._fold_splits: list[tuple[list[int], list[int]]] = []

    def _iter_kfold_splits(self) -> Iterator[tuple[list[int], list[int]]]:
        """Yield full-dataset train and validation indices for each fold.

        Yields:
            Tuples of ``(train_indices, valid_indices)`` for one fold.
        """
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)

        for _ in range(self.n_repeats):
            folds = np.array_split(np.random.permutation(indices), self.n_splits)
            for fold_idx, valid_indices in enumerate(folds):
                train_indices = np.concatenate(folds[:fold_idx] + folds[fold_idx + 1 :])
                yield train_indices.tolist(), valid_indices.tolist()

    def _fold_dataset(self, train_indices: list[int], valid_indices: list[int]) -> Dataset:
        """Return a dataset copy configured for one k-fold split.

        Args:
            train_indices: Training indices for this fold.
            valid_indices: Validation indices for this fold.

        Returns:
            A shallow copy of ``self.dataset`` with train and validation
            subsets set to the provided index lists.
        """
        dataset_fold = copy.copy(self.dataset)
        dataset_fold._subsets = [
            Subset(self.dataset, train_indices),
            Subset(self.dataset, valid_indices),
        ]
        return dataset_fold

    def _dataset_for_model(self, model_idx: int) -> Dataset:
        """Return the fold-specific dataset view for one model.

        Args:
            model_idx: Index of the model and corresponding fold split.

        Returns:
            A shallow dataset copy configured with that fold's train and
            validation subsets.
        """
        train_indices, valid_indices = self._fold_splits[model_idx]
        return self._fold_dataset(train_indices, valid_indices)

    def setup_models(self) -> None:
        """Instantiate one model for each fold across all repeats."""
        model_type = self.model_config.get_str("model_type")
        model_cls = ModelRegistry.get(model_type)

        self.models = []
        for model_idx in range(self.n_models):
            logging.info(f"Initializing k-fold model {model_idx + 1}/{self.n_models}: {model_type}")
            self.models.append(model_cls(**self.model_config.as_kwargs()))

    def init_model_weights(self) -> None:
        """Apply weight and bias initialization to every fold model."""
        if len(self.models) == 0:
            raise ValueError("Cannot initialize model weights because models are not initialized.")

        logging.info(f"Initializing weights with '{self.weight_init}' and bias with '{self.bias_init}'")
        for model_idx, model in enumerate(self.models):
            logging.info(f"Initializing k-fold model {model_idx + 1}/{self.n_models} weights.")
            model.init_weights(self.weight_init, self.bias_init, **self.weight_init_params.as_kwargs())

    def set_state_dicts(self, state_dicts: list[Mapping[str, Any]]) -> None:
        """Load one state dictionary into each fold model.

        Args:
            state_dicts: State dictionaries to load, one per fold model.

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
        """Instantiate one trainer per fold.

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
        self._fold_splits = list(self._iter_kfold_splits())
        self.trainers = []

        for model_idx, model in enumerate(self.models):
            logging.info(f"Initializing k-fold trainer {model_idx + 1}/{self.n_models}: {trainer_type}")
            dataset_model = self._dataset_for_model(model_idx)
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
        """Train all fold models and return them.

        Must be called after ``setup_trainers``.

        Returns:
            List of trained fold models.

        Raises:
            ValueError: If models or trainers are not initialized.
        """
        if len(self.models) == 0:
            raise ValueError("Cannot run training because models are not initialized.")
        if len(self.trainers) != len(self.models):
            raise ValueError("Cannot run training because trainers are not initialized for every model.")
        if self.checkpointer is None:
            raise ValueError("Cannot run training because checkpointer is not instantiated.")

        super().run_training()

        for fold_idx, trainer in enumerate(self.trainers):
            if trainer is None:
                raise ValueError("Cannot run training because trainers are not initialized for every model.")

            logging.info(f"Training k-fold model {fold_idx + 1}/{self.n_models}.")
            self.checkpointer.new_model()

            try:
                if self.tensorboard_dir is not None:
                    tb_logger.new_run(Path(self.tensorboard_dir) / f"fold_{fold_idx}")

                trainer.train()
            finally:
                tb_logger.close()

                self.models[fold_idx].to(torch.device("cpu"))
                self.trainers[fold_idx] = None

        logging.info("K-fold cross-validation finished.")
        return self.models

    def setup_inferencers(self, device: str | torch.device) -> None:
        """Instantiate the ensemble inferencer for all fold models.

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
        """Run aggregate inference with all k-fold models.

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
        """Return the model architecture signature.

        Returns:
            A ``Config`` representing the model signature.

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
                "n_splits": self.n_splits,
                "n_repeats": self.n_repeats,
            }
        )
        return signature
