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

"""K-fold cross-validation training and inference strategy for XANESNET."""

import copy
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model, ModelRegistry
from xanesnet.runners.inferencers import InferencerRegistry
from xanesnet.runners.trainers import TrainerRegistry
from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import tb_logger

from .base import Strategy
from .registry import StrategyRegistry


@StrategyRegistry.register("kfold")
class KFold(Strategy):
    """Repeated k-fold cross-validation strategy returning the best fold model.

    The strategy trains one model per fold on a shuffled partition of the full
    dataset. Each fold uses the holdout partition as validation during training.
    After all folds complete, the model with the lowest validation score is
    returned for inference.

    Args:
        strategy_type: Registry key identifying this strategy type.
        dataset: Dataset used for training or inference.
        model_config: Configuration for the model.
        encoding: Composed spectra encoding forwarded to the trainers and
            inferencer.
        weight_init: Weight initialization scheme name.
        weight_init_params: Additional weight-initializer parameters.
        bias_init: Bias initialization scheme name.
        n_splits: Number of folds per repeat.
        n_repeats: Number of times to repeat the k-fold split.
        seed: Random seed used to shuffle samples before splitting.
        checkpoint_dir: Directory for checkpoints, or ``None``.
        checkpoint_interval: Epoch interval between checkpoints, or ``None``.
        tensorboard_dir: Directory for TensorBoard event files, or ``None``.
        trainer_config: Trainer configuration for training mode.
        inferencer_config: Inferencer configuration for inference mode.
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
        n_splits: int = 3,
        n_repeats: int = 1,
        seed: int | None = None,
        trainer_config: Config | None = None,
        inferencer_config: Config | None = None,
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

        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}.")
        if n_repeats < 1:
            raise ValueError(f"n_repeats must be at least 1, got {n_repeats}.")
        if len(self.dataset) < n_splits:
            raise ValueError(
                f"Dataset has {len(self.dataset)} samples, but k-fold requires at least {n_splits}."
            )

        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed if seed is not None else np.random.default_rng().integers(0, 1000)
        self._rng = np.random.default_rng(self.seed)

        self.model: Model | None = None
        self.trainer: Any | None = None
        self.inferencer: Any | None = None
        self._device: str | torch.device | None = None

    def _iter_kfold_splits(self) -> Iterator[tuple[list[int], list[int]]]:
        """Yield train and validation index lists for each fold.

        Yields:
            Tuples of ``(train_indices, valid_indices)`` for one fold.
        """
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)

        for _ in range(self.n_repeats):
            shuffled = self._rng.permutation(indices)
            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
            fold_sizes[: n_samples % self.n_splits] += 1

            current = 0
            for fold_size in fold_sizes:
                test_indices = shuffled[current : current + fold_size]
                train_indices = np.concatenate([shuffled[:current], shuffled[current + fold_size :]])
                current += fold_size
                yield train_indices.tolist(), test_indices.tolist()

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

    def setup_models(self) -> None:
        """Instantiate a template model from ``model_config`` for signatures."""
        model_type = self.model_config.get_str("model_type")
        logging.info(f"Initializing k-fold model template: {model_type}")
        self.model = ModelRegistry.create(model_type, **self.model_config.as_kwargs())

    def init_model_weights(self) -> None:
        """Apply weight and bias initialization to the template model."""
        if self.model is None:
            raise ValueError("Cannot initialize model weights because the model is not initialized.")

        logging.info(f"Initializing weights with '{self.weight_init}' and bias with '{self.bias_init}'")
        self.model.init_weights(self.weight_init, self.bias_init, **self.weight_init_params.as_kwargs())

    def set_state_dicts(self, state_dicts: list[dict]) -> None:
        """Load model weights from the first entry of ``state_dicts``.

        Args:
            state_dicts: List of state dictionaries; only the first entry is
                used for the selected k-fold model.

        Raises:
            ValueError: If ``setup_models`` has not been called.
        """
        if self.model is None:
            raise ValueError("Cannot load state dicts because the model is not initialized.")

        self.model.load_state_dict(state_dicts[0])

    def setup_trainers(self, device: str | torch.device) -> None:
        """Store the training device; trainers are created per fold at runtime.

        Must be called after ``setup_models`` and ``setup_checkpointer``.

        Args:
            device: The device on which training will be performed.

        Raises:
            ValueError: If the model, trainer config, or checkpointer are not initialized.
        """
        if self.model is None:
            raise ValueError("Cannot setup trainers because the model is not initialized.")
        if self.trainer_config is None:
            raise ValueError("Can not setup trainers because there is no trainer config.")
        if self.checkpointer is None:
            raise ValueError("Can not setup trainers because checkpointer is not instantiated.")

        self._device = device
        self.trainer = None

    def run_training(self) -> list[Model]:
        """Train one model per fold and return the best-scoring model.

        Must be called after ``setup_trainers``.

        Returns:
            A single-element list containing the fold model with the lowest
            validation score.

        Raises:
            ValueError: If setup steps were not completed or no fold produced
                a usable validation score.
        """
        if self.model is None:
            raise ValueError("Cannot run training because the model is not initialized.")
        if self.trainer_config is None:
            raise ValueError("Cannot run training because there is no trainer config.")
        if self.checkpointer is None:
            raise ValueError("Cannot run training because checkpointer is not instantiated.")
        if self._device is None:
            raise ValueError("Cannot run training because trainers are not initialized.")

        super().run_training()

        model_type = self.model_config.get_str("model_type")
        model_cls = ModelRegistry.get(model_type)
        trainer_type = self.trainer_config.get_str("trainer_type")
        trainer_cls = TrainerRegistry.get(trainer_type)

        best_model: Model | None = None
        best_score = float("inf")
        valid_scores: list[float] = []
        n_folds = self.n_splits * self.n_repeats

        for fold_idx, (train_indices, valid_indices) in enumerate(self._iter_kfold_splits()):
            logging.info(f"Training k-fold model {fold_idx + 1}/{n_folds}.")
            self.checkpointer.new_model()

            model = model_cls(**self.model_config.as_kwargs())
            model.init_weights(self.weight_init, self.bias_init, **self.weight_init_params.as_kwargs())
            dataset_fold = self._fold_dataset(train_indices, valid_indices)
            trainer = trainer_cls(
                **self.trainer_config.as_kwargs(),
                dataset=dataset_fold,
                model=model,
                device=self._device,
                checkpointer=self.checkpointer,
                encoding=self.encoding,
            )

            try:
                if self.tensorboard_dir is not None:
                    tb_logger.new_run(Path(self.tensorboard_dir) / f"fold_{fold_idx}")

                score = trainer.train()
            finally:
                tb_logger.close()

            if score is None:
                logging.warning(f"Fold {fold_idx + 1} did not produce a validation score and will be skipped.")
                model.to(torch.device("cpu"))
                continue

            valid_scores.append(score)

            logging.info(f"Fold {fold_idx + 1} validation score: {score:.6f}")
            if score < best_score:
                logging.info(f"New best k-fold model found with validation score: {score:.6f}")
                best_score = score
                best_model = copy.deepcopy(model)

            model.to(torch.device("cpu"))

        if best_model is None:
            raise ValueError("K-fold training did not produce a model with a validation score.")

        logging.info("K-fold cross-validation finished.")
        if valid_scores:
            logging.info(
                f"Average validation score: {np.mean(valid_scores):.6f} +/- {np.std(valid_scores):.6f}"
            )

        self.model = best_model
        return [self.model]

    def setup_inferencers(self, device: str | torch.device) -> None:
        """Instantiate an inferencer for the selected k-fold model.

        Must be called after ``setup_models``.

        Args:
            device: The device on which inference will be performed.

        Raises:
            ValueError: If the model or inferencer config are not initialized.
        """
        if self.model is None:
            raise ValueError("Can not setup inferencers because the model is not initialized.")
        if self.inferencer_config is None:
            raise ValueError("Can not setup inferencers because there is no inferencer config.")

        inferencer_type = self.inferencer_config.get_str("inferencer_type")
        logging.info(f"Initializing inferencer: {inferencer_type}")

        inferencer = InferencerRegistry.create(
            inferencer_type,
            **self.inferencer_config.as_kwargs(),
            dataset=self.dataset,
            model=self.model,
            device=device,
            encoding=self.encoding,
        )

        self.inferencer = inferencer

    def run_inference(self, predictions_save_path: str | Path | None) -> None:
        """Run inference with the selected k-fold model.

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
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot retrieve signature.")

        return self.model.signature

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
                "seed": self.seed,
            }
        )
        return signature
