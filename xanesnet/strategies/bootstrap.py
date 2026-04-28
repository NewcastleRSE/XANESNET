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

"""Bootstrap ensemble strategy for XANESNET (placeholder)."""

from pathlib import Path

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.serialization.config import Config

from .base import Strategy
from .registry import StrategyRegistry


@StrategyRegistry.register("bootstrap")
class Bootstrap(Strategy):
    """Bootstrap ensemble strategy.

    Note:
        Not yet implemented.
        TODO: implement bootstrap sampling for model ensembling.

    Args:
        strategy_type: Strategy identifier.
        dataset: Dataset used for training or inference.
        model_config: Configuration for the model.
        weight_init: Weight initialization scheme name.
        weight_init_params: Additional weight-initializer parameters.
        bias_init: Bias initialization scheme name.
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
        weight_init: str,
        weight_init_params: Config,
        bias_init: str,
        checkpoint_dir: str | Path | None,
        checkpoint_interval: int | None,
        tensorboard_dir: str | Path | None,
        trainer_config: Config | None = None,
        inferencer_config: Config | None = None,
    ) -> None:
        """Initialize the placeholder bootstrap strategy."""
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

    def setup_models(self) -> None:
        """Raise because bootstrap model setup is not implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def init_model_weights(self) -> None:
        """Raise because bootstrap weight initialization is not implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def set_state_dicts(self, state_dicts: list[dict]) -> None:
        """Raise because bootstrap state-dict loading is not implemented.

        Args:
            state_dicts: State dictionaries that would be loaded into managed models.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def setup_trainers(self, device: str | torch.device) -> None:
        """Raise because bootstrap trainer setup is not implemented.

        Args:
            device: Target device for training.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def run_training(self) -> list[Model]:
        """Raise because bootstrap training is not implemented.

        Returns:
            Never returns normally.

        Raises:
            NotImplementedError: Always.
        """
        super().run_training()

        raise NotImplementedError("Not implemented!")  # TODO Implement

    def setup_inferencers(self, device: str | torch.device) -> None:
        """Raise because bootstrap inferencer setup is not implemented.

        Args:
            device: Target device for inference.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def run_inference(self, predictions_save_path: str | Path | None) -> None:
        """Raise because bootstrap inference is not implemented.

        Args:
            predictions_save_path: Destination directory for prediction output.

        Raises:
            NotImplementedError: Always.
        """
        super().run_inference(predictions_save_path)

        raise NotImplementedError("Not implemented!")  # TODO Implement

    @property
    def model_signature(self) -> Config:
        """Raise because bootstrap model signatures are not implemented.

        Returns:
            Never returns normally.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Not implemented!")  # TODO Implement

    @property
    def signature(self) -> Config:
        """Return the placeholder bootstrap strategy configuration.

        Returns:
            A ``Config`` containing the base strategy signature only.
        """
        signature = super().signature
        signature.update_with_dict({})
        return signature
