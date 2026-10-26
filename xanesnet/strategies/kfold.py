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

"""K-fold cross-validation strategy for XANESNET."""

import copy
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from xanesnet.datasets import Dataset
from xanesnet.models import Model, ModelRegistry
from xanesnet.runners.inferencers import InferencerRegistry
from xanesnet.runners.trainers import TrainerRegistry
from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import tb_logger
from xanesnet.utils.exceptions import ConfigError

from .base import Strategy
from .registry import StrategyRegistry


@StrategyRegistry.register("kfold")
class KFold(Strategy):
    """Repeated k-fold cross-validation ensemble training strategy.

    The strategy trains one model per fold on a shuffled partition of the full
    dataset; configured dataset splits are ignored. Each fold uses the holdout
    partition as validation during training. There are
    ``n_splits * n_repeats`` fold models in total. The folds are drawn once by
    ``setup_model_splits`` and can be restored unchanged, for example by
    distributed workers. All fold models share one encoding fitted on the full
    dataset, including every fold's training and validation samples.
    All fold models are retained and their predictions are aggregated by the
    ensemble inferencer.

    Args:
        strategy_type: Registry key identifying this strategy type.
        dataset: Dataset used for training or inference.
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

    def setup_model_splits(
        self,
        model_splits: list[list[list[int]]] | None = None,
    ) -> list[list[list[int]]]:
        """Draw or restore the train and validation indices of every fold.

        Args:
            model_splits: Previously drawn fold splits to restore, or ``None``
                to draw new folds from the full dataset.

        Raises:
            ConfigError: If new folds are requested with more folds than
                dataset samples.
            ValueError: If restored splits do not contain one entry per fold
                model.
        """
        if model_splits is None:
            if self.n_splits > len(self.dataset):
                raise ConfigError(
                    f"KFold n_splits={self.n_splits} exceeds the dataset size {len(self.dataset)}; "
                    "every fold requires at least one validation sample."
                )
            model_splits = [list(fold_split) for fold_split in self._iter_kfold_splits()]
        return super().setup_model_splits(model_splits)

    def _dataset_for_encoding(self) -> Dataset:
        """Use the union of all fold train and validation samples for fitting."""
        encoding_dataset = copy.copy(self.dataset)
        all_indices = list(range(len(self.dataset)))
        encoding_dataset._subsets = [Subset(self.dataset, all_indices)]
        return encoding_dataset

    def setup_models(self, model_config: Config) -> None:
        """Instantiate one model for each fold across all repeats."""
        model_type = model_config.get_str("model_type")
        model_cls = ModelRegistry.get(model_type)

        self.models = []
        for model_idx in range(self.n_models):
            logging.info(f"Initializing k-fold model {model_idx + 1}/{self.n_models}: {model_type}")
            self.models.append(model_cls(**model_config.as_kwargs()))

    def setup_trainers(self, device: str | torch.device) -> None:
        """Instantiate one trainer per fold.

        Each trainer receives the fold's dataset view and shared encoding.
        Must be called after ``setup_models``, ``setup_checkpointer``,
        ``setup_model_splits``, and ``setup_encoding``.

        Args:
            device: The device on which training will be performed.

        Raises:
            ValueError: If models, trainer config, checkpointer, fold splits,
                or encoding are not initialized.
        """
        if not self.models:
            raise ValueError("Cannot setup trainers because models are not initialized.")
        if self.trainer_config is None:
            raise ValueError("Cannot setup trainers because there is no trainer config.")
        if self.checkpointer is None:
            raise ValueError("Cannot setup trainers because checkpointer is not instantiated.")
        if self.model_splits is None:
            raise ValueError("Cannot setup trainers because fold splits are not initialized.")
        if self.encoding is None:
            raise ValueError("Cannot setup trainers because encoding is not initialized.")

        trainer_type = self.trainer_config.get_str("trainer_type")
        trainer_cls = TrainerRegistry.get(trainer_type)
        self.trainers = []

        for model_idx, model in enumerate(self.models):
            logging.info(f"Initializing k-fold trainer {model_idx + 1}/{self.n_models}: {trainer_type}")
            dataset_model = self.model_datasets[model_idx]
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
        if not self.models:
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

        Must be called after ``setup_encoding``, ``setup_models``, and
        ``set_state_dicts``.

        Args:
            device: The device on which inference will be performed.

        Raises:
            ValueError: If models, inferencer config, or encoding are
                not initialized.
        """
        if not self.models:
            raise ValueError("Cannot setup inferencers because models are not initialized.")
        if self.inferencer_config is None:
            raise ValueError("Cannot setup inferencers because there is no inferencer config.")
        if self.encoding is None:
            raise ValueError("Cannot setup inferencers because encoding is not initialized.")

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
    def signature(self) -> Config:
        """Return the strategy configuration as a ``Config``.

        Returns:
            A ``Config`` capturing the strategy configuration and the fitted
            split parameters used to restore the strategy.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "n_splits": self.n_splits,
                "n_repeats": self.n_repeats,
            }
        )
        return signature
