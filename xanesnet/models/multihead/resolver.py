# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

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
