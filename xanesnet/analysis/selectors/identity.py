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

"""Selector that yields each prediction sample unchanged."""

from collections.abc import Iterator

from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample

from .base import Selector
from .registry import SelectorRegistry


@SelectorRegistry.register("all")
@SelectorRegistry.register("none")
class IdentitySelector(Selector):
    """Yield all source samples unchanged.

    The registered ``all`` and ``none`` selector names both use this identity behavior.

    Args:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader to select samples from.
    """

    def __init__(
        self,
        selector_type: str,
        data_source: PredictionReader,
    ) -> None:
        """Initialize an identity selector."""
        super().__init__(selector_type, data_source)

    def __iter__(self) -> Iterator[PredictionSample]:
        """Yield all samples from ``data_source``.

        Returns:
            Iterator over prediction samples.
        """
        yield from self.data_source
