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

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample


class Selector(ABC, Iterable[PredictionSample]):
    """
    Base class for selectors.
    """

    def __init__(
        self,
        selector_type: str,
        data_source: PredictionReader,
    ) -> None:
        self.selector_type = selector_type
        self.data_source = data_source

    @abstractmethod
    def __iter__(self) -> Iterator[PredictionSample]:
        """
        Return a fresh iterator that applies the selection logic.
        """
        ...
