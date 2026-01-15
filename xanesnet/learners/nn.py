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
from xanesnet.models import Model

from .base import Learner
from .registry import LearnerRegistry


@LearnerRegistry.register("nnlearner")
class NNLearner(Learner):
    def __init__(
        self,
        learner_type: str,
        params: dict,
        dataset: Dataset,
        model: Model,
    ):
        super().__init__(learner_type, params, dataset, model)
