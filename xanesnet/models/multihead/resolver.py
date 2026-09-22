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

"""Automatic configuration resolvers for multi-head models."""

from typing import Any, cast

import torch

from xanesnet.datasets import Dataset, DescriptorMultiheadDataset
from xanesnet.serialization.auto_config.registries import ModelAutoResolver
from xanesnet.serialization.config import ConfigRaw


def _resolve_multihead_fields(inputs: dict[str, Any], target: torch.Tensor, dataset: Dataset) -> ConfigRaw:
    """Resolve shared input and equal per-head output dimensions.

    Args:
        inputs: Representative model inputs prepared by descriptor-based
            multi-head batch processors.
        target: Representative encoded target tensor used to resolve the
            common output width.
        dataset: Prepared descriptor multi-head dataset used to determine the
            number of heads.

    Returns:
        Resolved ``in_size`` and ``out_size`` model configuration fields.

    """
    multihead_dataset = cast(DescriptorMultiheadDataset, dataset)
    return {
        "in_size": int(inputs["x"].shape[-1]),
        "out_size": [int(target.shape[-1])] * multihead_dataset.num_heads,
    }


@ModelAutoResolver.register("mh_mlp")
def resolve_mh_mlp(inputs: dict[str, Any], target: torch.Tensor, dataset: Dataset) -> ConfigRaw:
    """Resolve automatic dimensions for ``MultiHeadMLP``.

    Args:
        inputs: Representative model inputs from the batch processor.
        target: Representative encoded target tensor used to resolve the
            common output width.
        dataset: Prepared descriptor multi-head dataset used to determine the
            number of heads.

    Returns:
        Resolved multi-head model fields.
    """
    return _resolve_multihead_fields(inputs, target, dataset)


@ModelAutoResolver.register("mh_cnn")
def resolve_mh_cnn(inputs: dict[str, Any], target: torch.Tensor, dataset: Dataset) -> ConfigRaw:
    """Resolve automatic dimensions for ``MultiHeadCNN``.

    Args:
        inputs: Representative model inputs from the batch processor.
        target: Representative encoded target tensor used to resolve the
            common output width.
        dataset: Prepared descriptor multi-head dataset used to determine the
            number of heads.

    Returns:
        Resolved multi-head model fields.
    """
    return _resolve_multihead_fields(inputs, target, dataset)
