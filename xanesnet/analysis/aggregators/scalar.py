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

import logging
from typing import Any, Iterable

import numpy as np

from .base import Aggregator
from .registry import AggregatorRegistry


@AggregatorRegistry.register("scalar")
class ScalarAggregator(Aggregator):
    """
    Computes statistics (mean, std, min, max, median, percentiles) for each value.
    """

    def __init__(
        self,
        aggregator_type: str,
        percentiles: list[float] | None = None,
    ) -> None:
        super().__init__(aggregator_type)

        self.percentiles = percentiles if percentiles is not None else [25, 50, 75]

    def aggregate(self, selector: Iterable[dict[str, Any]], per_sample_results: list[dict[str, Any]]) -> dict[str, Any]:
        if not per_sample_results:
            logging.warning("No per-sample results to aggregate")
            return {}

        # Collect all value names from first sample
        if len(per_sample_results) == 0:
            return {}

        # Get all unique keys across all samples (excluding sample_id)
        all_keys = set()
        for sample_dict in per_sample_results:
            all_keys.update(k for k in sample_dict.keys() if k != "sample_id")

        aggregated = {}

        # Aggregate each value
        for value_name in all_keys:
            # Collect all values for this key
            values = []
            for sample_dict in per_sample_results:
                if value_name in sample_dict:
                    values.append(sample_dict[value_name])

            if not values:
                continue

            # Convert to numpy array and compute statistics
            try:
                values_arr = np.array(values)
                aggregated[value_name] = {
                    "mean": float(np.mean(values_arr)),
                    "std": float(np.std(values_arr)),
                    "min": float(np.min(values_arr)),
                    "max": float(np.max(values_arr)),
                    "median": float(np.median(values_arr)),
                }

                # Add percentiles
                for p in self.percentiles:
                    aggregated[value_name][f"p{p}"] = float(np.percentile(values_arr, p))
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not aggregate '{value_name}': {e}")
                continue

        return aggregated
