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

"""Aggregator that summarizes vector sample and collector values into per-element statistics."""

import logging
from itertools import repeat
from typing import Any

import numpy as np

from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from ..selectors import Selector
from ..utils import as_float_vector, iter_vector_items
from .base import Aggregator, AggregatorResult
from .registry import AggregatorRegistry


@AggregatorRegistry.register("vector")
class VectorAggregator(Aggregator):
    """Compute per-element summary statistics for all vector sample and collector values.

    The ``prediction`` and ``target`` spectra are excluded.

    Requires:
        Vector collector output (optional).

    Args:
        aggregator_type: Registered aggregator name from the analysis configuration.
        percentiles: Percentile levels in ``[0, 100]``.
    """

    def __init__(self, aggregator_type: str, percentiles: list[float]) -> None:
        """Initialize a vector summary aggregator."""
        super().__init__(aggregator_type)

        self.percentiles = percentiles

    def aggregate(self, selector: Selector, per_sample_values: JSONLStream | None, index: int) -> AggregatorResult:
        """Aggregate vector values into per-element summary statistics.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            per_sample_values: Collector result stream aligned with ``selector``, or ``None`` when
                no collectors were configured.
            index: Zero-based aggregator index from the analysis configuration.

        Returns:
            Aggregated per-element statistics grouped by input key.
        """
        values_by_key: dict[str, list[np.ndarray]] = {}

        for sample, record in zip(selector, per_sample_values if per_sample_values is not None else repeat({})):
            self._collect_vectors(sample, values_by_key)
            self._collect_vectors(record, values_by_key)

        if not values_by_key:
            logging.info("      No vector values found, skipping.")

        data = {name: self._compute_stats(values) for name, values in values_by_key.items()}
        return AggregatorResult(aggregator_type=self.aggregator_type, aggregator_index=index, data=data)

    @staticmethod
    def _collect_vectors(sample: dict[str, Any] | PredictionSample, target: dict[str, list[np.ndarray]]) -> None:
        """Append vector values from ``sample`` into ``target`` by key.

        Sample metadata such as ``sample_id`` and ``target_site_index`` and the
        ``prediction`` and ``target`` spectra are skipped so they are not
        summarized as if they were measurements.

        Args:
            sample: Prediction sample or collector output mapping.
            target: Mutable mapping from value key to accumulated vector values.
        """
        for key, value in iter_vector_items(sample):
            target.setdefault(key, []).append(as_float_vector(value))

    def _compute_stats(self, values: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Compute per-element summary statistics for vector values.

        Args:
            values: Non-empty list of equal-length vectors.

        Returns:
            Statistics dictionary containing ``mean``, ``std``, ``min``, ``max``, ``median``, and
            configured percentile keys, each with shape ``(n_elements,)``.
        """
        stack = np.stack(values)
        stats = {
            "mean": stack.mean(axis=0),
            "std": stack.std(axis=0),
            "min": stack.min(axis=0),
            "max": stack.max(axis=0),
            "median": np.median(stack, axis=0),
        }
        for p in self.percentiles:
            stats[f"p{p}"] = np.percentile(stack, p, axis=0)
        return stats

    @property
    def signature(self) -> Config:
        """Return the vector aggregator signature.

        Returns:
            Configuration values needed to recreate this aggregator.
        """
        signature = super().signature
        signature.update_with_dict({"percentiles": self.percentiles})
        return signature
