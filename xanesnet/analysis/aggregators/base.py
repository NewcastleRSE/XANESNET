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
from dataclasses import dataclass
from typing import Any

from xanesnet.serialization.jsonl_stream import JSONLStream

from ..selectors import Selector


@dataclass(frozen=True)
class AggregatorResult:
    """
    Result from a single aggregator.

    Attributes:
        aggregator_type: The registered name of the aggregator (e.g. "scalar").
        aggregator_index: Position in the configured aggregators list.
        data: The actual aggregation output dict.
    """

    aggregator_type: str
    aggregator_index: int
    data: dict[str, Any]


class Aggregator(ABC):
    """
    Base class for aggregators.

    Aggregators compute statistics or summaries from collected per-sample values.
    """

    def __init__(
        self,
        aggregator_type: str,
    ) -> None:
        self.aggregator_type = aggregator_type

    @abstractmethod
    def aggregate(self, selector: Selector, per_sample_values: JSONLStream, index: int) -> AggregatorResult:
        """
        Aggregate per-sample values into combined values.
        """
        ...
