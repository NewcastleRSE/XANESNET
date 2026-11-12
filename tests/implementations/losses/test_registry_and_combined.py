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

"""Unit tests for the loss registry and ``CombinedLoss``."""

import pytest
import torch

from xanesnet.losses import CombinedLoss, L1Loss, Loss, LossRegistry, MSELoss

PREDS = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
TARGETS = torch.tensor([[1.0, 4.0, 3.0], [1.0, 1.0, 2.0]])


@pytest.mark.parametrize("key", ["mse", "l1", "bce", "emd"])
def test_parameterless_losses_are_created_through_the_registry(key):
    loss = LossRegistry.create(key, loss_type=key)

    assert isinstance(loss, Loss)
    assert loss.loss_type == key
    assert loss(PREDS, torch.ones_like(PREDS).clamp(0, 1)).dim() == 0


def test_lookup_ignores_case():
    assert LossRegistry.get("MSE") is LossRegistry.get("mse")


###############################################################################
################################# COMBINED LOSS ###############################
###############################################################################


def test_combined_loss_is_the_weighted_sum_of_the_component_loss_maps():
    combined = CombinedLoss([MSELoss("mse"), L1Loss("l1")], weights=[1.0, 3.0])

    expected_map = 0.25 * (PREDS - TARGETS) ** 2 + 0.75 * (PREDS - TARGETS).abs()

    torch.testing.assert_close(combined(PREDS, TARGETS, reduction="none"), expected_map)
    torch.testing.assert_close(combined(PREDS, TARGETS), expected_map.mean())


def test_weights_are_normalised_to_sum_to_one():
    scaled = CombinedLoss([MSELoss("mse"), L1Loss("l1")], weights=[10.0, 30.0])
    reference = CombinedLoss([MSELoss("mse"), L1Loss("l1")], weights=[1.0, 3.0])

    torch.testing.assert_close(scaled(PREDS, TARGETS), reference(PREDS, TARGETS))
    assert scaled.weights.sum().item() == pytest.approx(1.0)


def test_a_single_component_reproduces_that_loss():
    combined = CombinedLoss([MSELoss("mse")], weights=[5.0])
    torch.testing.assert_close(combined(PREDS, TARGETS), MSELoss("mse")(PREDS, TARGETS))


def test_weights_follow_the_module_and_are_part_of_its_state():
    combined = CombinedLoss([MSELoss("mse"), L1Loss("l1")], weights=[1.0, 1.0])
    assert "weights" in combined.state_dict()


@pytest.mark.parametrize(
    ("losses", "weights", "message"),
    [
        ([], [], "at least one"),
        ([MSELoss("mse")], [1.0, 2.0], "Length mismatch"),
    ],
)
def test_invalid_construction_is_rejected(losses, weights, message):
    with pytest.raises(ValueError, match=message):
        CombinedLoss(losses, weights)


def test_unsupported_reduction_is_rejected():
    combined = CombinedLoss([MSELoss("mse")], weights=[1.0])
    with pytest.raises(ValueError, match="Unsupported reduction"):
        combined(PREDS, TARGETS, reduction="sum")
