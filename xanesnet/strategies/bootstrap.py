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

from pathlib import Path

import torch

from xanesnet.datasets import Dataset
from xanesnet.models import Model
from xanesnet.serialization.config import Config

from .base import Strategy
from .registry import StrategyRegistry


@StrategyRegistry.register("bootstrap")
class Bootstrap(Strategy):

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
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def init_model_weights(self) -> None:
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def set_state_dicts(self, state_dicts: list[dict]) -> None:
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def setup_trainers(self, device: str | torch.device) -> None:
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def run_training(self) -> list[Model]:
        super().run_training()

        raise NotImplementedError("Not implemented!")  # TODO Implement

    def setup_inferencers(self, device: str | torch.device) -> None:
        raise NotImplementedError("Not implemented!")  # TODO Implement

    def run_inference(self, predictions_save_path: str | Path | None) -> None:
        super().run_inference(predictions_save_path)

        raise NotImplementedError("Not implemented!")  # TODO Implement

    @property
    def model_signature(self) -> Config:
        raise NotImplementedError("Not implemented!")  # TODO Implement

    @property
    def signature(self) -> Config:
        signature = super().signature
        signature.update_with_dict({})
        return signature
