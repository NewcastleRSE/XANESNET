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
from typing import List

from xanesnet.datasets import Dataset
from xanesnet.models import Model


class Strategy(ABC):

    def __init__(
        self,
        strategy_type: str,
        dataset: Dataset,
        model_config: dict,
        trainer_config: dict = None,
        inferencer_config: dict = None,
        params: dict = {},
    ):
        self.strategy_type = strategy_type
        self.dataset = dataset
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.inferencer_config = inferencer_config
        self.params = params

        if self.trainer_config is None and self.inferencer_config is None:
            raise ValueError("Either trainer_config or inferencer_config must be provided.")

    @abstractmethod
    def setup_models(self):
        """
        Instantiates the models that will be used for training.
        This method should be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def init_model_weights(self):
        """
        Initialises the model weights.
        This method should be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def set_state_dicts(self, state_dicts: List[dict]):
        """
        Sets the state dictionaries for the models.
        This method should be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def setup_trainers(self, device: str):
        """
        Instantiates the trainers that will be used for training.
        This method should be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def run_training(self) -> List[Model]:
        """
        Starts training with strategy and returns a list of trained models.
        This method should be implemented by all subclasses.

        Returns a list of trained models.
        """

        logging.info("Start strategy...")

    @abstractmethod
    def setup_inferencers(self, device: str):
        """
        Instantiates the inferencers that will be used for inference.
        This method should be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def run_inference(self):
        """
        Starts inference with strategy.
        This method should be implemented by all subclasses.
        """

        logging.info("Start inference...")

    @property
    @abstractmethod
    def model_signature(self) -> dict:
        """
        Return model signature as a dictionary.
        This method should be implemented by all subclasses.
        """
        pass

    @property
    def signature(self) -> dict:
        """
        Returns strategy signature as a dictionary.
        """

        return {
            "strategy_type": self.strategy_type,
            # TODO more signature entries?
        }
