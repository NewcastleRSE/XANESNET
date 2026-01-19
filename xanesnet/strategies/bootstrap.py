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

from xanesnet.datasets import Dataset

from .base import Strategy
from .registry import StrategyRegistry


@StrategyRegistry.register("bootstrap")
class Bootstrap(Strategy):

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
        pass  # TODO Implement

    def setup_learners(self, device: str):
        pass  # TODO Implement

    def run_training(self):
        super().run_training()

        return []  # TODO Implement
