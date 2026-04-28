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

"""Selector that samples fixed prediction indices with Bernoulli trials."""

import random
from collections.abc import Iterator

from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample

from .base import Selector
from .registry import SelectorRegistry


@SelectorRegistry.register("random")
class BernoulliSelector(Selector):
    """Select a random fixed subset of samples using Bernoulli sampling.

    Args:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader to select samples from.
        p: Probability of retaining each sample. Must be in the inclusive range ``[0, 1]``.
    """

    def __init__(
        self,
        selector_type: str,
        data_source: PredictionReader,
        p: float,
    ) -> None:
        """Initialize the selector and draw the retained sample indices.

        Raises:
            ValueError: If ``p`` is outside the inclusive range ``[0, 1]``.
        """
        super().__init__(selector_type, data_source)

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")

        self.p = p
        self._selected_indices: list[int] = [i for i in range(len(data_source)) if random.random() < p]

    def __iter__(self) -> Iterator[PredictionSample]:
        """Yield the prediction samples retained at initialization time.

        Returns:
            Iterator over selected prediction samples.
        """
        for i in self._selected_indices:
            yield self.data_source[i]
