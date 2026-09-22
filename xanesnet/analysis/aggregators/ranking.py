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

"""Aggregator that ranks selected samples by a collected scalar value."""

from typing import Any

import numpy as np

from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample
from xanesnet.utils.exceptions import ConfigError

from ..selectors import Selector
from ..utils import as_float_vector, is_scalar_value
from .base import Aggregator, AggregatorResult
from .registry import AggregatorRegistry


@AggregatorRegistry.register("ranking")
class RankingAggregator(Aggregator):
    """Rank selected samples by a collected scalar value.

    Samples are ranked in ascending order by their ``sort_key`` scalar and
    split into best and worst ``percent`` groups; per ``group_keys`` vector the
    best and worst group mean and standard deviation are computed.

    Requires:
        Per-sample ranking values: provided by a scalar collector emitting ``sort_key``.

    Args:
        aggregator_type: Registered aggregator name from the analysis configuration.
        sort_key: Scalar collector key used to rank samples. Must be produced
            by a configured collector.
        percent: Percentage of samples in the best and worst groups.
        group_keys: Vector keys (from samples or collector output) for which
            the best and worst group mean and standard deviation are computed.
    """

    def __init__(
        self,
        aggregator_type: str,
        sort_key: str,
        percent: float,
        group_keys: list[str],
    ) -> None:
        """Initialize a ranking aggregator."""
        super().__init__(aggregator_type)
        self.sort_key = sort_key
        self.percent = percent
        self.group_keys = group_keys

    def aggregate(self, selector: Selector, per_sample_values: JSONLStream | None, index: int) -> AggregatorResult:
        """Rank the selected samples into best and worst percent groups.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            per_sample_values: Collector result stream aligned with ``selector``; must be provided
                so the configured ``sort_key`` can be read.
            index: Zero-based aggregator index from the analysis configuration.

        Returns:
            Ranking output with ``sample_ids`` and parallel
            ``target_site_indices``, ``values``, ``order``, ``best_indices``
            and ``worst_indices``, ``best_sample_ids`` and
            ``worst_sample_ids`` with parallel target-site indices, the
            ``n_samples`` and ``n_tail`` counts, the configured ``percent`` and
            ``sort_key``, and one
            ``best_<key>_mean``/``best_<key>_std``/``worst_<key>_mean``/
            ``worst_<key>_std`` entry per configured ``group_keys`` key.

        Raises:
            ConfigError: If no collector stream is available or the ``sort_key``
                is missing or non-scalar for any sample.
        """
        if per_sample_values is None:
            raise ConfigError(f"Ranking aggregator requires a collector stream to read sort key '{self.sort_key}'.")

        aligned: list[tuple[PredictionSample, dict[str, Any]]] = list(zip(selector, per_sample_values))
        if not aligned:
            return AggregatorResult(aggregator_type=self.aggregator_type, aggregator_index=index, data={})

        sample_ids: list[str] = []
        target_site_indices: list[int | None] = []
        values: list[float] = []
        for sample, record in aligned:
            sample_id = sample["sample_id"]
            value = record.get(self.sort_key)
            if not is_scalar_value(value):
                raise ConfigError(
                    f"Sort key '{self.sort_key}' is missing or not a scalar for sample '{sample_id}'. "
                    "Configure a scalar collector that produces this key."
                )
            sample_ids.append(sample_id)
            target_site_indices.append(sample.get("target_site_index"))
            values.append(float(value))

        n_samples = len(sample_ids)
        n_tail = max(1, int(np.ceil(n_samples * self.percent / 100.0)))
        order = [int(i) for i in np.argsort(values)]
        best_indices = order[:n_tail]
        worst_indices = order[-n_tail:]

        data: dict[str, Any] = {
            "sample_ids": sample_ids,
            "target_site_indices": target_site_indices,
            "values": values,
            "order": order,
            "best_indices": best_indices,
            "worst_indices": worst_indices,
            "best_sample_ids": [sample_ids[i] for i in best_indices],
            "worst_sample_ids": [sample_ids[i] for i in worst_indices],
            "best_target_site_indices": [target_site_indices[i] for i in best_indices],
            "worst_target_site_indices": [target_site_indices[i] for i in worst_indices],
            "n_samples": n_samples,
            "n_tail": n_tail,
            "percent": self.percent,
            "sort_key": self.sort_key,
        }
        data.update(self._group_stats(aligned, best_indices, worst_indices))
        return AggregatorResult(aggregator_type=self.aggregator_type, aggregator_index=index, data=data)

    def _group_stats(
        self,
        aligned: list[tuple[PredictionSample, dict[str, Any]]],
        best_indices: list[int],
        worst_indices: list[int],
    ) -> dict[str, np.ndarray]:
        """Compute best/worst group mean and std for each configured group key.

        Args:
            aligned: ``(sample, record)`` pairs in selection order.
            best_indices: Indices of the best samples.
            worst_indices: Indices of the worst samples.

        Returns:
            Mapping from ``<direction>_<key>_<stat>`` to per-channel arrays.

        Raises:
            ConfigError: If a group key is missing from both the sample and the
                collector record.
        """
        stats: dict[str, np.ndarray] = {}
        for key in self.group_keys:
            arrays: list[np.ndarray] = []
            for sample, record in aligned:
                raw = record.get(key)
                if raw is None:
                    raw = sample.get(key)
                if raw is None:
                    raise ConfigError(f"Group key '{key}' not found in sample or collector output.")
                arrays.append(as_float_vector(raw))
            stack = np.stack(arrays)
            for direction, indices in (("best", best_indices), ("worst", worst_indices)):
                group = stack[indices]
                stats[f"{direction}_{key}_mean"] = group.mean(axis=0)
                stats[f"{direction}_{key}_std"] = group.std(axis=0)
        return stats

    @property
    def signature(self) -> Config:
        """Return the ranking aggregator signature.

        Returns:
            Configuration values needed to recreate this aggregator.
        """
        signature = super().signature
        signature.update_with_dict({"sort_key": self.sort_key, "percent": self.percent, "group_keys": self.group_keys})
        return signature
