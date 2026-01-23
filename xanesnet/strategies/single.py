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
from pathlib import Path
from typing import Any

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model, ModelRegistry
from xanesnet.runners.inferencers import InferencerRegistry
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
        weight_init: str,
        weight_init_params: dict[str, Any],
        bias_init: str,
        checkpoint_dir: str | Path | None,
        checkpoint_interval: int | None,
        trainer_config: dict[str, Any] | None = None,
        inferencer_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            strategy_type,
            dataset,
            model_config,
            weight_init,
            weight_init_params,
            bias_init,
            checkpoint_dir,
            checkpoint_interval,
            trainer_config,
            inferencer_config,
        )

    def setup_models(self) -> None:
        model_type = self.model_config["model_type"]
        logging.info(f"Initialising model: {model_type}")
        model = ModelRegistry.get(model_type)(**self.model_config)

        self.model = model

    def init_model_weights(self) -> None:
        # Intialise model weights
        logging.info(f"Initialising weights with '{self.weight_init}' and bias with '{self.bias_init}'")
        self.model.init_weights(self.weight_init, self.bias_init, **self.weight_init_params)

    def set_state_dicts(self, state_dicts: list[dict]) -> None:
        self.model.load_state_dict(state_dicts[0])

    def setup_trainers(self, device: str | torch.device) -> None:
        if self.trainer_config is None:
            raise ValueError("Can not setup trainers because there is no trainer config.")
        if self.checkpointer is None:
            raise ValueError("Can not setup trainers because checkpointer is not instantiated.")

        trainer_type = self.trainer_config["trainer_type"]

        logging.info(f"Initialising trainer: {trainer_type}")

        trainer = TrainerRegistry.get(trainer_type)(
            **self.trainer_config,
            dataset=self.dataset,
            model=self.model,
            device=device,
            checkpointer=self.checkpointer,
        )

        self.trainer = trainer

    def run_training(self) -> list[Model]:
        if self.trainer is None:
            raise ValueError("Cannot run training because the trainer is not initialised.")

        super().run_training()

        assert self.checkpointer is not None
        self.checkpointer.new_model()

        _ = self.trainer.train()  # TODO should we do something with the returned score?

        return [self.model]

    def setup_inferencers(self, device: str | torch.device) -> None:
        if self.inferencer_config is None:
            raise ValueError("Can not setup inferencers because there is no inferencer config.")

        inferencer_type = self.inferencer_config["inferencer_type"]

        logging.info(f"Initialising inferencer: {inferencer_type}")

        inferencer = InferencerRegistry.get(inferencer_type)(
            **self.inferencer_config,
            dataset=self.dataset,
            model=self.model,
            device=device,
        )

        self.inferencer = inferencer

    def run_inference(self) -> None:
        if self.inferencer is None:
            raise ValueError("Cannot run inference because the Inferencer is not initialised.")

        super().run_inference()

        _ = self.inferencer.infer()  # TODO should we do something with the returned score?

        return None

    @property
    def model_signature(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot retrieve signature.")

        return self.model.signature

    @property
    def signature(self) -> dict[str, Any]:
        """
        Returns strategy signature as a dictionary.
        """
        signature = super().signature
        signature.update({})
        return signature
