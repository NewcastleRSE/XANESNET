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

"""Aggregator that summarizes scalar values from samples and collectors."""

import logging
from itertools import repeat
from typing import Any

import numpy as np

from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from ..selectors import Selector
from ..utils import iter_scalar_items
from .base import Aggregator, AggregatorResult
from .registry import AggregatorRegistry


@AggregatorRegistry.register("scalar")
class ScalarAggregator(Aggregator):
    """Compute summary statistics for all scalar sample and collector values.

    Scalars sharing a key are collected over the selected samples and reduced
    to summary statistics.

    Requires:
        Scalar collector output (optional).

    Args:
        aggregator_type: Registered aggregator name from the analysis configuration.
        percentiles: Percentile levels in ``[0, 100]``.
    """

    def __init__(self, aggregator_type: str, percentiles: list[float]) -> None:
        """Initialize a scalar summary aggregator."""
        super().__init__(aggregator_type)

        self.percentiles = percentiles

    def aggregate(self, selector: Selector, per_sample_values: JSONLStream | None, index: int) -> AggregatorResult:
        """Aggregate scalar values into mean, spread, extrema, and percentile statistics.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            per_sample_values: Collector result stream aligned with ``selector``, or ``None`` when
                no collectors were configured.
            index: Zero-based aggregator index from the analysis configuration.

        Returns:
            Aggregated scalar statistics grouped by input key.
        """
        values_by_key: dict[str, list[float]] = {}

        for sample, record in zip(selector, per_sample_values if per_sample_values is not None else repeat({})):
            self._collect_scalars(sample, values_by_key)
            self._collect_scalars(record, values_by_key)

        if not values_by_key:
            logging.info("      No scalar values found, skipping.")

        data = {name: self._compute_stats(values) for name, values in values_by_key.items()}
        return AggregatorResult(aggregator_type=self.aggregator_type, aggregator_index=index, data=data)

    @staticmethod
    def _collect_scalars(sample: dict[str, Any] | PredictionSample, target: dict[str, list[float]]) -> None:
        """Append scalar values from ``sample`` into ``target`` by key.

        Sample metadata such as ``sample_id`` and ``target_site_index`` is
        skipped so it is not summarized as if it were a measurement.

        Args:
            sample: Prediction sample or collector output mapping.
            target: Mutable mapping from value key to accumulated scalar values.
        """
        for key, value in iter_scalar_items(sample):
            target.setdefault(key, []).append(float(value))

    def _compute_stats(self, values: list[float]) -> dict[str, float]:
        """Compute summary statistics for scalar values.

        Args:
            values: Non-empty list of scalar values.

        Returns:
            Statistics dictionary containing ``mean``, ``std``, ``min``, ``max``, ``median``, and
            configured percentile keys.
        """
        arr = np.array(values)
        stats = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }
        for p in self.percentiles:
            stats[f"p{p}"] = float(np.percentile(arr, p))
        return stats

    @property
    def signature(self) -> Config:
        """Return the scalar aggregator signature.

        Returns:
            Configuration values needed to recreate this aggregator.
        """
        signature = super().signature
        signature.update_with_dict({"percentiles": self.percentiles})
        return signature
