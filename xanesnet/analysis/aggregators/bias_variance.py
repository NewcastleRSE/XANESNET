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

"""Aggregator for the per-channel bias-variance decomposition of the MSE."""

from typing import Any

import numpy as np

from xanesnet.serialization.jsonl_stream import JSONLStream

from ..selectors import Selector
from .base import Aggregator, AggregatorResult
from .registry import AggregatorRegistry


@AggregatorRegistry.register("bias_variance")
class BiasVarianceAggregator(Aggregator):
    """Split the per-channel MSE into its squared bias and variance components.

    The identity ``mse = bias2 + variance`` holds exactly per channel.

    Args:
        aggregator_type: Registered aggregator name from the analysis configuration.
    """

    def __init__(self, aggregator_type: str) -> None:
        """Initialize a bias-variance aggregator."""
        super().__init__(aggregator_type)

    def aggregate(self, selector: Selector, per_sample_values: JSONLStream | None, index: int) -> AggregatorResult:
        """Decompose the per-channel error of the selected samples.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            per_sample_values: Not used; only the selected samples are
                aggregated.
            index: Zero-based aggregator index from the analysis configuration.

        Returns:
            Per-channel ``mse``, ``bias``, ``bias2``, and ``variance`` curves
            plus the ``n_samples`` count. The data is empty when no samples are
            selected.
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
        residuals = preds - targets
        bias = residuals.mean(axis=0)
        bias2 = bias**2
        variance = ((residuals - bias) ** 2).mean(axis=0)

        data: dict[str, Any] = {
            "mse": bias2 + variance,
            "bias": bias,
            "bias2": bias2,
            "variance": variance,
            "n_samples": len(preds_list),
        }
        return AggregatorResult(aggregator_type=self.aggregator_type, aggregator_index=index, data=data)
