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

"""Automatic configuration resolvers for the DimeNet and DimeNet++ models."""

from typing import Any

import torch

from xanesnet.datasets import Dataset
from xanesnet.serialization.auto_config.registries import ModelAutoResolver
from xanesnet.serialization.config import ConfigRaw


@ModelAutoResolver.register("dimenet")
def resolve_dimenet(inputs: dict[str, Any], target: torch.Tensor, dataset: Dataset) -> ConfigRaw:
    """Resolve DimeNet output dimension.

    Args:
        inputs: Prepared model input dictionary.
        target: Prepared target tensor.
        dataset: Prepared dataset, unused by this resolver.

    Returns:
        Mapping with DimeNet automatic field ``out_channels``.
    """
    return {"out_channels": int(target.shape[-1])}


@ModelAutoResolver.register("dimenet++")
def resolve_dimenet_pp(inputs: dict[str, Any], target: torch.Tensor, dataset: Dataset) -> ConfigRaw:
    """Resolve DimeNet++ output dimension.

    Args:
        inputs: Prepared model input dictionary.
        target: Prepared target tensor.
        dataset: Prepared dataset, unused by this resolver.

    Returns:
        Mapping with DimeNet++ automatic field ``out_channels``.
    """
    return {"out_channels": int(target.shape[-1])}
