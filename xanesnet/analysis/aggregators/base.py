# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

"""Base aggregator interfaces and result dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from xanesnet.serialization.jsonl_stream import JSONLStream

from ..selectors import Selector


@dataclass(frozen=True)
class AggregatorResult:
    """Result from a single aggregator invocation.

    Attributes:
        aggregator_type: Registered name of the aggregator, e.g. ``"scalar"``.
        aggregator_index: Position in the configured aggregators list.
        data: Aggregation output keyed by value name.
    """

    aggregator_type: str
    aggregator_index: int
    data: dict[str, Any]


class Aggregator(ABC):
    """Base class for aggregation over selected samples and collector outputs.

    Args:
        aggregator_type: Registered aggregator name from the analysis configuration.

    Attributes:
        aggregator_type: Registered aggregator name from the analysis configuration.
    """

    def __init__(
        self,
        aggregator_type: str,
    ) -> None:
        """Initialize an aggregator instance."""
        self.aggregator_type = aggregator_type

    @abstractmethod
    def aggregate(self, selector: Selector, per_sample_values: JSONLStream, index: int) -> AggregatorResult:
        """Aggregate selected samples and per-sample collector values.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            per_sample_values: Collector result stream aligned with ``selector``.
            index: Zero-based aggregator index from the analysis configuration.

        Returns:
            Aggregated result for this selector and aggregator.
        """
        ...
