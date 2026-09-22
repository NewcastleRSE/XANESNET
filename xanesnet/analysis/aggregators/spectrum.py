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

"""Aggregator that summarizes prediction and target spectra over selected samples."""

from typing import Any

import numpy as np

from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream

from ..selectors import Selector
from .base import Aggregator, AggregatorResult
from .registry import AggregatorRegistry


@AggregatorRegistry.register("spectrum")
class SpectrumAggregator(Aggregator):
    """Compute per-channel summary statistics for the predicted and target spectra.

    Args:
        aggregator_type: Registered aggregator name from the analysis configuration.
        percentiles: Percentile levels in ``[0, 100]``.
    """

    def __init__(self, aggregator_type: str, percentiles: list[float]) -> None:
        """Initialize a spectrum aggregator."""
        super().__init__(aggregator_type)

        self.percentiles = percentiles

    def aggregate(self, selector: Selector, per_sample_values: JSONLStream | None, index: int) -> AggregatorResult:
        """Aggregate the selected spectra into per-channel summary statistics.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            per_sample_values: Not used; only the selected samples are
                aggregated.
            index: Zero-based aggregator index from the analysis configuration.

        Returns:
            Per-channel summary statistics for the predicted and target spectra
            (``mean``, ``std``, ``min``, ``max``, ``median``, and configured
            percentiles, prefixed with ``prediction_`` and ``target_``) plus the
            ``n_samples`` count. The data is empty when no samples are selected.
        """
        preds_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []

        for sample in selector:
            preds_list.append(np.asarray(sample["prediction"]).ravel())
            targets_list.append(np.asarray(sample["target"]).ravel())

        if not preds_list:
            return AggregatorResult(aggregator_type=self.aggregator_type, aggregator_index=index, data={})

        preds = np.stack(preds_list)
        targets = np.stack(targets_list)

        data: dict[str, Any] = {
            **{f"target_{key}": value for key, value in self._channel_stats(targets).items()},
            **{f"prediction_{key}": value for key, value in self._channel_stats(preds).items()},
            "n_samples": len(preds_list),
        }
        return AggregatorResult(aggregator_type=self.aggregator_type, aggregator_index=index, data=data)

    def _channel_stats(self, spectra: np.ndarray) -> dict[str, np.ndarray]:
        """Compute per-channel summary statistics for stacked spectra.

        Args:
            spectra: Stacked spectra with shape ``(n_samples, n_channels)``.

        Returns:
            Statistics dictionary containing ``mean``, ``std``, ``min``, ``max``,
            ``median``, and configured percentile keys, each with shape
            ``(n_channels,)``.
        """
        stats = {
            "mean": spectra.mean(axis=0),
            "std": spectra.std(axis=0),
            "min": spectra.min(axis=0),
            "max": spectra.max(axis=0),
            "median": np.median(spectra, axis=0),
        }
        for p in self.percentiles:
            stats[f"p{p}"] = np.percentile(spectra, p, axis=0)
        return stats

    @property
    def signature(self) -> Config:
        """Return the spectrum aggregator signature.

        Returns:
            Configuration values needed to recreate this aggregator.
        """
        signature = super().signature
        signature.update_with_dict({"percentiles": self.percentiles})
        return signature
