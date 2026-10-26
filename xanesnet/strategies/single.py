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

"""Single-model strategy for XANESNET."""

import logging
from pathlib import Path

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model, ModelRegistry
from xanesnet.runners.inferencers import InferencerRegistry
from xanesnet.runners.trainers import TrainerRegistry
from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import tb_logger

from .base import Strategy
from .registry import StrategyRegistry


@StrategyRegistry.register("single")
class Single(Strategy):
    """Single-model training and inference strategy.

    Trains or runs inference with exactly one model instance. This is the
    standard strategy for most XANESNET workflows.

    Args:
        strategy_type: Registry key identifying this strategy type.
        dataset: Dataset used for training or inference.
        weight_init: Weight initialization scheme name.
        weight_init_params: Additional parameters for the weight initializer.
        bias_init: Bias initialization scheme name.
        checkpoint_dir: Directory for checkpoints, or ``None`` to disable checkpointing.
        checkpoint_interval: Epoch interval for checkpoint saves, or ``None`` to disable interval-based checkpointing.
        tensorboard_dir: Directory for TensorBoard event files, or ``None`` to disable TensorBoard logging.
        trainer_config: Trainer configuration used for training mode.
        inferencer_config: Inferencer configuration used for inference mode.
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
        """Initialize the single-model strategy."""
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

    def setup_models(self, model_config: Config) -> None:
        """Instantiate the single model managed by this strategy."""
        model_type = model_config.get_str("model_type")
        logging.info(f"Initializing model: {model_type}")
        self.models = [ModelRegistry.create(model_type, **model_config.as_kwargs())]

    def setup_trainers(self, device: str | torch.device) -> None:
        """Instantiate one trainer for the strategy's single model.

        Must be called after ``setup_models``, ``setup_checkpointer``,
        ``setup_model_splits``, and ``setup_encoding``.

        Args:
            device: The device on which training will be performed.

        Raises:
            ValueError: If ``setup_models`` has not been called,
                ``trainer_config`` is ``None``, or the checkpointer has not
                been set up.
        """
        if not self.models:
            raise ValueError("Cannot setup trainers because the model is not initialized.")
        if self.trainer_config is None:
            raise ValueError("Cannot setup trainers because there is no trainer config.")
        if self.checkpointer is None:
            raise ValueError("Cannot setup trainers because checkpointer is not instantiated.")
        if self.encoding is None:
            raise ValueError("Cannot setup trainer because encoding is not initialized.")

        trainer_type = self.trainer_config.get_str("trainer_type")

        logging.info(f"Initializing trainer: {trainer_type}")

        trainer = TrainerRegistry.create(
            trainer_type,
            **self.trainer_config.as_kwargs(),
            dataset=self.model_datasets[0],
            model=self.models[0],
            device=device,
            checkpointer=self.checkpointer,
            encoding=self.encoding,
        )

        self.trainers = [trainer]

    def run_training(self) -> list[Model]:
        """Run the training loop and return the trained model.

        Must be called after ``setup_trainers``.

        Returns:
            A single-element list containing the trained model.

        Raises:
            ValueError: If ``setup_models`` or ``setup_trainers`` has not been called.
        """
        if not self.trainers or not self.models:
            raise ValueError("Cannot run training because the model or trainer is not initialized.")
        assert self.trainers[0] is not None

        super().run_training()

        assert self.checkpointer is not None
        self.checkpointer.new_model()

        try:
            if self.tensorboard_dir is not None:
                tb_logger.new_run(self.tensorboard_dir)

            self.trainers[0].train()
        finally:
            tb_logger.close()

        return self.models

    def setup_inferencers(self, device: str | torch.device) -> None:
        """Instantiate the inferencer for the strategy's single model.

        Must be called after ``setup_models`` and ``setup_encoding``.

        Args:
            device: The device on which inference will be performed.

        Raises:
            ValueError: If ``setup_models`` has not been called or
                ``inferencer_config`` is ``None``.
        """
        if not self.models:
            raise ValueError("Cannot setup inferencers because the model is not initialized.")
        if self.inferencer_config is None:
            raise ValueError("Cannot setup inferencers because there is no inferencer config.")
        if self.encoding is None:
            raise ValueError("Cannot setup inferencer because encoding is not initialized.")

        inferencer_type = self.inferencer_config.get_str("inferencer_type")

        logging.info(f"Initializing inferencer: {inferencer_type}")

        inferencer = InferencerRegistry.create(
            inferencer_type,
            **self.inferencer_config.as_kwargs(),
            dataset=self.dataset,
            model=self.models[0],
            device=device,
            encoding=self.encoding,
        )

        self.inferencer = inferencer

    def run_inference(self, predictions_save_path: str | Path | None) -> None:
        """Run inference and optionally save predictions.

        Must be called after ``setup_inferencers``.

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
            A ``Config`` capturing the strategy configuration.
        """
        signature = super().signature
        return signature
