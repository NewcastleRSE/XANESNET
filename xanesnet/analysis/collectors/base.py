# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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
#
# Citations:
#   ...

"""Base collector interface for per-sample analysis values."""

from abc import ABC, abstractmethod
from typing import Any

from xanesnet.serialization.config import Config
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

    @property
    def signature(self) -> Config:
        """Return the collector signature.

        Returns:
            Configuration values needed to recreate this collector.
        """
        return Config({"collector_type": self.collector_type})

    def __str__(self) -> str:
        """Return the short display label of this collector."""
        return self.collector_type

    def __repr__(self) -> str:
        """Return a detailed representation of this collector."""
        args = ", ".join(f"{key}={value!r}" for key, value in self.signature.as_dict().items())
        return f"{type(self).__name__}({args})"
