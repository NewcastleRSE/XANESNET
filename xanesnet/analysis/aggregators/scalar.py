"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Any, TypeGuard

import numpy as np

from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from ..selectors import Selector
from .base import Aggregator, AggregatorResult
from .registry import AggregatorRegistry


def _is_scalar(value: Any) -> TypeGuard[int | float | np.integer | np.floating]:
    """Check if a value is a scalar int or float (excludes bools)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (np.integer, np.floating)):
        return True
    return False


@AggregatorRegistry.register("scalar")
class ScalarAggregator(Aggregator):
    """
    Computes statistics (mean, std, min, max, median, percentiles) for all scalar values.
    """

    def __init__(
        self,
        aggregator_type: str,
        percentiles: list[float] | None = None,
    ) -> None:
        super().__init__(aggregator_type)

        self.percentiles = percentiles if percentiles is not None else [25, 50, 75]

    def aggregate(self, selector: Selector, per_sample_values: JSONLStream, index: int) -> AggregatorResult:
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
        """
        Extract all scalar float/int values from a sample dict.
        """
        for key, value in sample.items():
            if _is_scalar(value):
                target.setdefault(key, []).append(float(value))

    def _compute_stats(self, values: list[float]) -> dict[str, float]:
        """
        Compute summary statistics for a list of scalar values.
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
