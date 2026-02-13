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

import random
from collections.abc import Iterator

from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample

from .base import Selector
from .registry import SelectorRegistry


@SelectorRegistry.register("random")
class BernoulliSelector(Selector):
    """
    Randomly select a subset of samples using Bernoulli sampling.
    """

    def __init__(
        self,
        selector_type: str,
        data_source: PredictionReader,
        p: float,
    ) -> None:
        super().__init__(selector_type, data_source)

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")

        self.p = p
        self._selected_indices: list[int] = [i for i in range(len(data_source)) if random.random() < p]

    def __iter__(self) -> Iterator[PredictionSample]:
        for i in self._selected_indices:
            yield self.data_source[i]
