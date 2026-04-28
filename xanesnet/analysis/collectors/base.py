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

"""Base collector interface for per-sample analysis values."""

from abc import ABC, abstractmethod
from typing import Any

from xanesnet.serialization.prediction_readers import PredictionSample


class Collector(ABC):
    """Base class for per-sample analysis collectors.

    Args:
        collector_type: Registered collector name from the analysis configuration.

    Attributes:
        collector_type: Registered collector name from the analysis configuration.
    """

    def __init__(
        self,
        collector_type: str,
    ) -> None:
        """Initialize a collector instance."""
        self.collector_type = collector_type

    @abstractmethod
    def process(self, sample: PredictionSample) -> dict[str, Any]:
        """Process a single prediction sample.

        Args:
            sample: Prediction sample containing at least ``prediction`` and ``target`` arrays.

        Returns:
            Mapping of string keys to JSON-serializable collected values.
        """
        ...
