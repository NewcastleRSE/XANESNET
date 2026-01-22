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
from typing import Any

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model, ModelRegistry
from xanesnet.runners.trainers import TrainerRegistry

from .base import Strategy
from .registry import StrategyRegistry


@StrategyRegistry.register("single")
class Single(Strategy):

    def __init__(
        self,
        strategy_type: str,
        dataset: Dataset,
        model_config: dict[str, Any],
        trainer_config: dict[str, Any] | None = None,
        inferencer_config: dict[str, Any] | None = None,
        params: dict[str, Any] = {},
    ) -> None:
        super().__init__(strategy_type, dataset, model_config, trainer_config, inferencer_config, params)

    def setup_models(self) -> None:
        model_type = self.model_config["model_type"]
        logging.info(f"Initialising model: {model_type}")
        model_params = self.model_config.get("params", {})
        model = ModelRegistry.get(model_type)(model_type=model_type, **model_params)

        self.model = model

    def init_model_weights(self) -> None:
        # Intialise model weights
        weights_params = self.model_config.get("weights_params", {})
        weights_init = self.model_config.get("weights_init", {}).get("weights", "default")
        bias_init = self.model_config.get("weights_init", {}).get("bias", "zeros")
        logging.info(f"Initialising weights with '{weights_init}' and bias with '{bias_init}'")
        self.model.init_weights(weights_init, bias_init, **weights_params)

    def set_state_dicts(self, state_dicts: list[dict]) -> None:
        self.model.load_state_dict(state_dicts[0])

    def setup_trainers(self, device: str | torch.device) -> None:
        if self.trainer_config is None:
            raise ValueError("Can not setup trainers because there is no trainer config.")

        trainer_type = self.trainer_config["trainer_type"]

        logging.info(f"Initialising trainer: {trainer_type}")

        trainer = TrainerRegistry.get(trainer_type)(
            **self.trainer_config,
            dataset=self.dataset,
            model=self.model,
            device=device,
        )

        self.trainer = trainer

    def run_training(self) -> list[Model]:
        super().run_training()

        _ = self.trainer.train()  # TODO should we do something with the returned score?

        return [self.model]

    def setup_inferencers(self, device: str | torch.device) -> None:
        if self.inferencer_config is None:
            raise ValueError("Can not setup inferencers because there is no inferencer config.")

        raise NotImplementedError("Not implemented!")  # TODO Implement

    def run_inference(self) -> None:
        super().run_inference()

        raise NotImplementedError("Not implemented!")  # TODO Implement

    @property
    def model_signature(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot retrieve signature.")

        return self.model.signature
