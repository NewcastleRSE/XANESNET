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

"""Base selector interface for analysis sample selection."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample


class Selector(ABC, Iterable[PredictionSample]):
    """Base class for iterable prediction sample selectors.

    Args:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader to select samples from.

    Attributes:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader to select samples from.
    """

    def __init__(
        self,
        selector_type: str,
        data_source: PredictionReader,
    ) -> None:
        """Initialize a selector instance."""
        self.selector_type = selector_type
        self.data_source = data_source

    @abstractmethod
    def __iter__(self) -> Iterator[PredictionSample]:
        """Return a fresh iterator that applies the selector logic.

        Returns:
            Iterator over selected prediction samples.
        """
        ...
