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

"""Single-model training and inference strategy for XANESNET."""

import logging
from pathlib import Path
from typing import Any

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
        model_config: Configuration for the managed model.
        weight_init: Weight initialization scheme name.
        weight_init_params: Additional parameters for the weight initializer.
        bias_init: Bias initialization scheme name.
        checkpoint_dir: Directory for checkpoints, or ``None`` to disable
            checkpointing.
        checkpoint_interval: Epoch interval for checkpoint saves, or
            ``None`` to disable interval-based checkpointing.
        tensorboard_dir: Directory for TensorBoard event files, or
            ``None`` to disable TensorBoard logging.
        trainer_config: Trainer configuration used for training mode.
        inferencer_config: Inferencer configuration used for inference mode.
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
        """Initialize the single-model strategy."""
        super().__init__(
            strategy_type,
            dataset,
            model_config,
            weight_init,
            weight_init_params,
            bias_init,
            checkpoint_dir,
            checkpoint_interval,
            tensorboard_dir,
            trainer_config,
            inferencer_config,
        )

        self.model: Model | None = None
        self.trainer: Any | None = None
        self.inferencer: Any | None = None

    def setup_models(self) -> None:
        """Instantiate a single model from ``model_config`` and store it as ``self.model``."""
        model_type = self.model_config.get_str("model_type")
        logging.info(f"Initializing model: {model_type}")
        model = ModelRegistry.get(model_type)(**self.model_config.as_kwargs())

        self.model = model

    def init_model_weights(self) -> None:
        """Apply weight and bias initialization to ``self.model``."""
        if self.model is None:
            raise ValueError("Cannot initialize model weights because the model is not initialized.")

        logging.info(f"Initializing weights with '{self.weight_init}' and bias with '{self.bias_init}'")
        self.model.init_weights(self.weight_init, self.bias_init, **self.weight_init_params.as_kwargs())

    def set_state_dicts(self, state_dicts: list[dict]) -> None:
        """Load model weights from the first entry of ``state_dicts``.

        Args:
            state_dicts: List of state dictionaries; only the first entry is
                used for the single model.

        Raises:
            ValueError: If ``setup_models`` has not been called.
        """
        if self.model is None:
            raise ValueError("Cannot load state dicts because the model is not initialized.")

        self.model.load_state_dict(state_dicts[0])

    def setup_trainers(self, device: str | torch.device) -> None:
        """Instantiate a trainer for ``self.model`` and store it as ``self.trainer``.

        Must be called after ``setup_models`` and ``setup_checkpointer``.

        Args:
            device: The device on which training will be performed.

        Raises:
            ValueError: If ``setup_models`` has not been called,
                ``trainer_config`` is ``None``, or the checkpointer has not
                been set up.
        """
        if self.model is None:
            raise ValueError("Cannot setup trainers because the model is not initialized.")
        if self.trainer_config is None:
            raise ValueError("Can not setup trainers because there is no trainer config.")
        if self.checkpointer is None:
            raise ValueError("Can not setup trainers because checkpointer is not instantiated.")

        trainer_type = self.trainer_config.get_str("trainer_type")

        logging.info(f"Initializing trainer: {trainer_type}")

        trainer = TrainerRegistry.get(trainer_type)(
            **self.trainer_config.as_kwargs(),
            dataset=self.dataset,
            model=self.model,
            device=device,
            checkpointer=self.checkpointer,
        )

        self.trainer = trainer

    def run_training(self) -> list[Model]:
        """Run the training loop and return the trained model.

        Must be called after ``setup_trainers``.

        Returns:
            A single-element list containing the trained model.

        Raises:
            ValueError: If ``setup_models`` or ``setup_trainers`` has not been called.
        """
        if self.trainer is None:
            raise ValueError("Cannot run training because the trainer is not initialized.")
        if self.model is None:
            raise ValueError("Cannot run training because the model is not initialized.")

        super().run_training()

        assert self.checkpointer is not None
        self.checkpointer.new_model()

        try:
            if self.tensorboard_dir is not None:
                tb_logger.new_run(self.tensorboard_dir)

            self.trainer.train()
        finally:
            tb_logger.close()

        return [self.model]

    def setup_inferencers(self, device: str | torch.device) -> None:
        """Instantiate an inferencer for ``self.model`` and store it as ``self.inferencer``.

        Must be called after ``setup_models``.

        Args:
            device: The device on which inference will be performed.

        Raises:
            ValueError: If ``setup_models`` has not been called or
                ``inferencer_config`` is ``None``.
        """
        if self.model is None:
            raise ValueError("Can not setup inferencers because the model is not initialized.")
        if self.inferencer_config is None:
            raise ValueError("Can not setup inferencers because there is no inferencer config.")

        inferencer_type = self.inferencer_config.get_str("inferencer_type")

        logging.info(f"Initializing inferencer: {inferencer_type}")

        inferencer = InferencerRegistry.get(inferencer_type)(
            **self.inferencer_config.as_kwargs(),
            dataset=self.dataset,
            model=self.model,
            device=device,
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
    def model_signature(self) -> Config:
        """Return ``self.model``'s configuration signature.

        Returns:
            A ``Config`` representing the model's signature.

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
            A ``Config`` capturing the strategy and model configuration.
        """
        signature = super().signature
        signature.update_with_dict({})
        return signature
