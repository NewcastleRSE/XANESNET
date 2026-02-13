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

from collections.abc import Iterator

from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample

from .base import Selector
from .registry import SelectorRegistry


@SelectorRegistry.register("index_list")
class IndexSelector(Selector):
    """
    Select samples by indices.
    """

    def __init__(
        self,
        selector_type: str,
        data_source: PredictionReader,
        indices: list[int],
    ) -> None:
        super().__init__(selector_type, data_source)

        if any(i < 0 for i in indices):
            raise ValueError("IndexSelector does not support negative indices")

        self.indices = set(indices)

    def __iter__(self) -> Iterator[PredictionSample]:
        for idx, sample in enumerate(self.data_source):
            if idx in self.indices:
                yield sample
