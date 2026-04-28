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

"""Dataclasses shared across analysis pipeline stages."""

from dataclasses import dataclass, field
from typing import Any

from xanesnet.serialization.jsonl_stream import JSONLStream

from .aggregators import AggregatorResult
from .selectors import Selector


@dataclass
class AnalysisResults:
    """Bundle of outputs produced by the analysis pipeline.

    Attributes:
        selectors: Selector instances grouped by prediction reader and selector index.
        collector_results: Per-sample collector output streams grouped like ``selectors``.
        aggregator_results: Aggregation outputs grouped by prediction reader, selector, and
            aggregator.
        selectors_config: Selector configuration dictionaries in pipeline order.
        collectors_config: Collector configuration dictionaries in pipeline order.
        aggregators_config: Aggregator configuration dictionaries in pipeline order.
    """

    selectors: list[list[Selector]]
    collector_results: list[list[JSONLStream]]
    aggregator_results: list[list[list[AggregatorResult]]]
    selectors_config: list[dict[str, Any]] = field(default_factory=list)
    collectors_config: list[dict[str, Any]] = field(default_factory=list)
    aggregators_config: list[dict[str, Any]] = field(default_factory=list)
