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

"""Auto-resolver registries for model and encoding configuration."""

from collections.abc import Callable
from typing import Any

import torch

from xanesnet.datasets import Dataset
from xanesnet.utils.registry import Registry

from ..config import ConfigRaw
from .statistics import SpectralStatisticsCollector

ModelResolver = Callable[[dict[str, Any], torch.Tensor, Dataset], ConfigRaw]
"""Model auto-resolver signature: ``(inputs, target, dataset) -> resolved_fields``.

The first argument is the model-specific input dictionary produced by the
batch processor.  The second is the encoded target tensor whose final
dimension typically determines the output size.  The third is the prepared
dataset, which provides dataset-level information for fields that cannot be
inferred from one input and target sample.  Returns a
:data:`~xanesnet.serialization.config.ConfigRaw` of resolved model
fields.
"""

EncodingResolver = Callable[[ConfigRaw, SpectralStatisticsCollector], ConfigRaw]
"""Encoding auto-resolver signature: ``(item, statistics) -> resolved_fields``.

The first argument is the raw encoding configuration dictionary.  The
second is a
:class:`~xanesnet.serialization.auto_config.statistics.SpectralStatisticsCollector`
populated from the training dataset.  Returns a
:data:`~xanesnet.serialization.config.ConfigRaw` of resolved encoding
fields.
"""

ModelAutoResolver: Registry[ModelResolver] = Registry("model auto-resolver", normalize_key=str.lower)
"""Registry of model-specific automatic field resolvers.

Keys are lower-case model type strings (e.g. ``"mlp"``, ``"schnet"``).
Register with::

    @ModelAutoResolver.register("my_model")
    def _resolve_my_model(
        inputs: dict[str, Any], target: torch.Tensor, dataset: Dataset
    ) -> ConfigRaw:
        ...
"""

EncodingAutoResolver: Registry[EncodingResolver] = Registry("encoding auto-resolver", normalize_key=str.lower)
"""Registry of encoding-specific automatic field resolvers.

Keys are lower-case encoding type strings (e.g. ``"gaussian"``, ``"z_score"``).
Register with::

    @EncodingAutoResolver.register("my_encoding")
    def _resolve_my_encoding(item: ConfigRaw, statistics: SpectralStatisticsCollector) -> ConfigRaw:
        ...
"""
