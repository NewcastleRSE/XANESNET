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

"""Aggregator that summarizes scalar values from samples and collectors."""

from typing import Any, TypeGuard

import numpy as np

from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from ..selectors import Selector
from .base import Aggregator, AggregatorResult
from .registry import AggregatorRegistry


def _is_scalar(value: Any) -> TypeGuard[int | float | np.integer | np.floating]:
    """Return whether ``value`` is a non-boolean integer or floating scalar.

    Args:
        value: Candidate value from a prediction sample or collector output.

    Returns:
        ``True`` when ``value`` is a Python or NumPy integer/floating scalar, excluding booleans.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (np.integer, np.floating)):
        return True
    return False


@AggregatorRegistry.register("scalar")
class ScalarAggregator(Aggregator):
    """Compute summary statistics for all scalar sample and collector values.

    Args:
        aggregator_type: Registered aggregator name from the analysis configuration.
        percentiles: Percentiles to compute. Values use NumPy percentile units, where ``0`` is the
            minimum and ``100`` is the maximum. Defaults to ``[25, 50, 75]``.
    """

    def __init__(
        self,
        aggregator_type: str,
        percentiles: list[float] | None = None,
    ) -> None:
        """Initialize a scalar summary aggregator."""
        super().__init__(aggregator_type)

        self.percentiles = percentiles if percentiles is not None else [25, 50, 75]

    def aggregate(self, selector: Selector, per_sample_values: JSONLStream, index: int) -> AggregatorResult:
        """Aggregate scalar values into mean, spread, extrema, and percentile statistics.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            per_sample_values: Collector result stream aligned with ``selector``.
            index: Zero-based aggregator index from the analysis configuration.

        Returns:
            Aggregated scalar statistics grouped by input key.

        Raises:
            ValueError: If no scalar values are present in either source.
        """
        values_by_key: dict[str, list[float]] = {}

        for sample in selector:
            self._collect_scalars(sample, values_by_key)

        for raw_sample in per_sample_values:
            self._collect_scalars(raw_sample, values_by_key)

        if not values_by_key:
            raise ValueError(f"ScalarAggregator: No scalar values found for selector {selector} at index {index}.")

        data = {name: self._compute_stats(values) for name, values in values_by_key.items()}
        result = AggregatorResult(
            aggregator_type=self.aggregator_type,
            aggregator_index=index,
            data=data,
        )
        return result

    @staticmethod
    def _collect_scalars(sample: dict[str, Any] | PredictionSample, target: dict[str, list[float]]) -> None:
        """Append scalar values from ``sample`` into ``target`` by key.

        Args:
            sample: Prediction sample or collector output mapping.
            target: Mutable mapping from value key to accumulated scalar values.
        """
        for key, value in sample.items():
            if _is_scalar(value):
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
