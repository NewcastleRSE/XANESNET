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

"""Dataclasses shared across analysis pipeline stages."""

from dataclasses import dataclass, field

from xanesnet.serialization.jsonl_stream import JSONLStream

from .aggregators import AggregatorResult
from .selectors import Selector
from .utils import one_line_label


@dataclass(frozen=True)
class MethodLabel:
    """Display labels for one prediction reader and selector pair.

    A "method" is one such pair: the predictions of one inference run viewed
    through one selector.

    Attributes:
        lines: Stacked label lines, the reader name followed by the selector
            description.
        dir_name: Output directory or file stem for this pair.
    """

    lines: list[str]
    dir_name: str

    @property
    def joined(self) -> str:
        """Return the stacked label lines joined into a single line.

        Returns:
            Display label with non-empty parts separated by ``" | "``.
        """
        return one_line_label(self.lines)


@dataclass
class AnalysisResults:
    """Bundle of outputs produced by the analysis pipeline.

    All three result lists are indexed prediction-reader first, then selector,
    so ``results.selectors[reader_idx][sel_idx]`` and
    ``results.collector_results[reader_idx][sel_idx]`` describe the same method.

    Attributes:
        selectors: Selector instances grouped by prediction reader and selector index.
        collector_results: Per-sample collector output streams grouped like ``selectors``.
        aggregator_results: Aggregation outputs grouped by prediction reader, selector, and
            aggregator.
        prediction_names: Display name per prediction reader.
    """

    selectors: list[list[Selector]]
    collector_results: list[list[JSONLStream]]
    aggregator_results: list[list[list[AggregatorResult]]]
    prediction_names: list[str] = field(default_factory=list)

    def collector_stream(self, reader_idx: int, sel_idx: int) -> JSONLStream | None:
        """Return the collector stream of one method, if collectors ran.

        Args:
            reader_idx: Zero-based prediction reader index.
            sel_idx: Zero-based selector index within the reader.

        Returns:
            Collector output stream aligned with the selector, or ``None``
            when no collectors were configured.
        """
        if not self.collector_results:
            return None
        return self.collector_results[reader_idx][sel_idx]

    def aggregation(self, reader_idx: int, sel_idx: int, aggregator_type: str) -> AggregatorResult:
        """Return the result of one aggregator type for one method.

        Args:
            reader_idx: Zero-based prediction reader index.
            sel_idx: Zero-based selector index within the reader.
            aggregator_type: Registered aggregator name to find.

        Returns:
            Matching aggregation result.

        Raises:
            StopIteration: If no result has the requested aggregator type.
        """
        return next(
            result
            for result in self.aggregator_results[reader_idx][sel_idx]
            if result.aggregator_type == aggregator_type
        )

    def method_label(self, reader_idx: int, sel_idx: int) -> MethodLabel:
        """Return the display labels of one method.

        Args:
            reader_idx: Zero-based prediction reader index.
            sel_idx: Zero-based selector index.

        Returns:
            Stacked label lines and output directory name.
        """
        selector = self.selectors[reader_idx][sel_idx]
        return MethodLabel(
            lines=[self.prediction_names[reader_idx], str(selector)],
            dir_name=f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{selector.selector_type}",
        )
