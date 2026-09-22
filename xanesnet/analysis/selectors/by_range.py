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

"""Selector that keeps samples from a half-open index range."""

from collections.abc import Iterator

from xanesnet.serialization.config import Config
from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample

from .base import Selector
from .registry import SelectorRegistry


@SelectorRegistry.register("index_range")
class IndexRangeSelector(Selector):
    """Select samples from ``start`` up to, but excluding, ``end``.

    Args:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader to select samples from.
        start: Inclusive zero-based start index.
        end: Exclusive zero-based end index. ``None`` keeps samples through the source end.
    """

    def __init__(self, selector_type: str, data_source: PredictionReader, start: int, end: int | None) -> None:
        """Initialize a half-open index range selector."""
        super().__init__(selector_type, data_source)

        self.start = start
        self.end = end

    def __iter__(self) -> Iterator[PredictionSample]:
        """Yield samples in the configured half-open index interval.

        Returns:
            Iterator over selected prediction samples.
        """
        for idx, sample in enumerate(self.data_source):
            if idx < self.start:
                continue
            if self.end is not None and idx >= self.end:
                break
            yield sample

    def __len__(self) -> int:
        """Return the number of selected samples.

        Returns:
            Number of selected prediction samples.
        """
        if self.end is None:
            return max(0, len(self.data_source) - self.start)
        return max(0, min(self.end, len(self.data_source)) - self.start)

    @property
    def signature(self) -> Config:
        """Return the selector signature.

        Returns:
            Configuration values needed to recreate this selector.
        """
        signature = super().signature
        signature.update_with_dict({"start": self.start, "end": self.end})
        return signature
