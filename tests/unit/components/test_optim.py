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

"""Unit tests for the XANESNET optimizer wrappers (signature, registry creation, argument forwarding)."""

import pytest
import torch

from xanesnet.components import Optimizer, OptimizerRegistry
from xanesnet.serialization.config import Config


@pytest.fixture
def model() -> torch.nn.Module:
    return torch.nn.Linear(3, 1)


@pytest.mark.parametrize("optimizer_type", OptimizerRegistry.list())
def test_registry_creates_an_optimizer_that_records_its_signature(optimizer_type, model):
    optimizer = OptimizerRegistry.create(optimizer_type, optimizer_type=optimizer_type, params=model.parameters(), lr=0.1)

    assert isinstance(optimizer, Optimizer)
    assert optimizer.torch_optimizer is optimizer
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert isinstance(optimizer.signature, Config)
    assert optimizer.signature.as_dict() == {"optimizer": optimizer_type}


def test_lookup_ignores_case_but_the_signature_keeps_the_users_spelling(model):
    optimizer = OptimizerRegistry.create("ADAM", optimizer_type="ADAM", params=model.parameters(), lr=0.01)

    assert type(optimizer) is OptimizerRegistry.get("adam")
    assert optimizer.signature.as_dict() == {"optimizer": "ADAM"}


def test_extra_keyword_arguments_reach_the_torch_optimizer(model):
    optimizer = OptimizerRegistry.create("sgd", optimizer_type="sgd", params=model.parameters(), lr=0.1, momentum=0.9)
    assert optimizer.param_groups[0]["momentum"] == 0.9
