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

"""Unit tests for the weight regularizers."""

import pytest
import torch

from xanesnet.regularizers import Regularizer, RegularizerRegistry


@pytest.fixture
def model() -> torch.nn.Module:
    layer = torch.nn.Linear(2, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
        layer.bias.copy_(torch.tensor([0.5, -0.5]))
    return layer


def create(key: str, **kwargs) -> Regularizer:
    return RegularizerRegistry.create(key, regularizer_type=key, **kwargs)


def test_l1_is_the_weighted_sum_of_absolute_parameter_values(model):
    assert create("l1", weight=0.1)(model).item() == pytest.approx(0.1 * (1 + 2 + 3 + 4 + 0.5 + 0.5))


def test_l2_is_the_weighted_sum_of_squared_parameter_values(model):
    assert create("l2", weight=0.1)(model).item() == pytest.approx(0.1 * (1 + 4 + 9 + 16 + 0.25 + 0.25))


@pytest.mark.parametrize("key", ["l1", "l2"])
def test_the_penalty_is_differentiable_with_respect_to_the_parameters(model, key):
    create(key, weight=1.0)(model).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("key", ["no", "none"])
def test_the_no_op_regularizer_adds_no_penalty(model, key):
    penalty = create(key, weight=5.0)(model)

    assert penalty.dim() == 0
    assert penalty.item() == 0.0
    assert penalty.dtype == next(model.parameters()).dtype


def test_the_no_op_regularizer_has_a_default_weight():
    assert RegularizerRegistry.create("no", regularizer_type="no").weight == 1.0


@pytest.mark.parametrize("key", ["l1", "l2", "no", "none"])
def test_registered_regularizers_record_their_type_and_weight(key):
    regularizer = create(key, weight=0.3)

    assert isinstance(regularizer, Regularizer)
    assert regularizer.regularizer_type == key
    assert regularizer.weight == 0.3
