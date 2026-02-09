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

    def aggregate(
        self,
        selector: Iterable[dict[str, Any]],
        per_sample_values: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        values_by_key: dict[str, list[Any]] = {}
        seen_any = False

        for sample_dict in per_sample_values:
            seen_any = True
            for key, value in sample_dict.items():
                if key == "sample_id":
                    continue
                values_by_key.setdefault(key, []).append(value)

        if not seen_any:
            logging.warning("No per-sample results to aggregate")
            return {}

        aggregated: dict[str, Any] = {}

        # Aggregate each value
        for value_name, values in values_by_key.items():
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
