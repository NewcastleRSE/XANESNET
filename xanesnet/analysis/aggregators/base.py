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
from typing import Any, Iterable


class Aggregator(ABC):
    """
    Base class for result aggregation modules.

    Aggregators compute statistics or summaries from per-sample results.
    They also have access to the selector (data source) for re-iteration if needed.
    """

    def __init__(
        self,
        aggregator_type: str,
    ) -> None:
        self.aggregator_type = aggregator_type

    @abstractmethod
    def aggregate(self, selector: Iterable[dict[str, Any]], per_sample_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregate per-sample results into summary statistics.

        Args:
            selector: The selector (data source) for this selection, can be iterated if needed
            per_sample_results: List of per-sample value dicts, one per sample

        Returns:
            Dictionary of aggregated results
        """
        ...
