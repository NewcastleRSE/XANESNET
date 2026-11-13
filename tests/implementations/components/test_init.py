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

"""Unit tests for the weight and bias initializer registries."""

import pytest
import torch

from xanesnet.components import BiasInitRegistry, WeightInitRegistry

ALL_INITIALIZERS = [("weight", key) for key in WeightInitRegistry.list()] + [
    ("bias", key) for key in BiasInitRegistry.list()
]
REGISTRIES = {"weight": WeightInitRegistry, "bias": BiasInitRegistry}


@pytest.mark.parametrize(("kind", "key"), ALL_INITIALIZERS)
def test_every_initializer_returns_the_tensor_it_was_given(kind, key):
    """The ``InitFn`` contract: initialize in place and return the same tensor object."""
    tensor = torch.ones(4, 4)
    assert REGISTRIES[kind].get(key)(tensor) is tensor


def test_default_initializer_leaves_the_tensor_untouched_and_accepts_keyword_arguments():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected = tensor.clone()

    assert WeightInitRegistry.get("default")(tensor, a=1.0, mode="fan_in") is tensor
    assert torch.equal(tensor, expected)


@pytest.mark.parametrize(("registry", "key"), [(WeightInitRegistry, "Xavier_Uniform"), (BiasInitRegistry, "ZEROS")])
def test_lookup_ignores_case(registry, key):
    assert registry.get(key) is registry.get(key.lower())
