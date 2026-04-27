"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from xanesnet.checkpointing import Checkpointer
from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.serialization.config import Config


class Strategy(ABC):
    """
    Abstract base class for strategies.
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
        """
        Instantiates the models that will be used for training.
        """
        ...

    @abstractmethod
    def init_model_weights(self) -> None:
        """
        Initialises the model weights.
        """
        ...

    @abstractmethod
    def set_state_dicts(self, state_dicts: list[dict]) -> None:
        """
        Sets the state dictionaries for the models.
        """
        ...

    @abstractmethod
    def setup_trainers(self, device: str | torch.device) -> None:
        """
        Instantiates the trainers that will be used for training.
        """
        ...

    @abstractmethod
    def run_training(self) -> list[Model]:
        """
        Starts training with strategy and returns a list of trained models.

        Returns a list of trained models.
        """
        logging.info("Start strategy...")
        return []

    @abstractmethod
    def setup_inferencers(self, device: str | torch.device) -> None:
        """
        Instantiates the inferencers that will be used for inference.
        """
        ...

    @abstractmethod
    def run_inference(self, predictions_save_path: str | Path | None) -> None:
        """
        Starts inference with strategy.
        """
        logging.info("Start strategy...")

    def setup_checkpointer(self) -> None:
        """
        Instantiates the Checkpointer.
        """
        self.checkpointer = Checkpointer(self.checkpoint_dir, self.checkpoint_interval, self.model_signature)

    @property
    @abstractmethod
    def model_signature(self) -> Config:
        """
        Returns model signature as a dictionary.
        """
        ...

    @property
    @abstractmethod
    def signature(self) -> Config:
        """
        Returns strategy signature as a dictionary.
        """
        signature = Config(
            {
                "strategy_type": self.strategy_type,
            }
        )
        return signature
