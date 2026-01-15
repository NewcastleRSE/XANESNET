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

from xanesnet.datasets import Dataset
from xanesnet.learners import LearnerRegistry
from xanesnet.models import ModelRegistry, PretrainedModels
from xanesnet.utils.io import load_pretrained_model

from .base import Strategy
from .registry import StrategyRegistry


@StrategyRegistry.register("single")
class Single(Strategy):

    def __init__(
        self,
        strategy_type: str,
        dataset: Dataset,
        model_config: dict,
        learner_config: dict,
        params: dict = {},
    ):
        super().__init__(strategy_type, dataset, model_config, learner_config, params)

    def setup_models(self):
        model_type = self.model_config["model_type"]

        if hasattr(PretrainedModels, model_type):
            # TODO -----------------------------------------------------
            # TODO check pretrained model loading
            # TODO also check if descriptors are loaded correctly
            logging.info(f"Loading pretrained model: {model_type}")
            model_params = self.model_config.get("params", {})
            model = load_pretrained_model(model_type, **model_params)
            # TODO -----------------------------------------------------
        else:
            logging.info(f"Initialising model: {model_type}")
            model_params = self.model_config.get("params", {})
            model = ModelRegistry.get(model_type)(model_type=model_type, **model_params, **self.dataset.metadata)

            # Intialise model weights
            weights_params = self.model_config.get("weights_params", {})
            weights_init = self.model_config.get("weights_init", {}).get("weights", "default")
            bias_init = self.model_config.get("weights_init", {}).get("bias", "zeros")
            logging.info(f"Initialising weights with '{weights_init}' and bias with '{bias_init}'")
            model.init_weights(weights_init, bias_init, **weights_params)

        self.model = model

    def setup_learners(self):
        learner_type = self.learner_config["learner_type"]

        logging.info(f"Initialising learner: {learner_type}")

        learner = LearnerRegistry.get(learner_type)(**self.learner_config, dataset=self.dataset, model=self.model)

        self.learner = learner

    def run_training(self):
        super().run_training()

        _ = self.learner.train()  # TODO should we do something with the returned score?

        return [self.model]
