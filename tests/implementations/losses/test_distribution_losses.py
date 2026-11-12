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

"""Unit tests for the ``emd`` (earth mover's distance) and ``wcc`` (weighted cross-correlation) losses."""

import pytest
import torch

from xanesnet.losses import EMDLoss, WCCLoss


def spike(position: int, width: int = 21, height: float = 1.0) -> torch.Tensor:
    signal = torch.zeros(1, width)
    signal[0, position] = height
    return signal


###############################################################################
##################################### EMD #####################################
###############################################################################


def test_emd_is_the_summed_absolute_difference_of_the_cumulative_distributions():
    preds = torch.tensor([[1.0, 0.0, 0.0]])
    targets = torch.tensor([[0.0, 0.0, 1.0]])

    loss = EMDLoss("emd")

    torch.testing.assert_close(loss(preds, targets, reduction="none"), torch.tensor([[1.0, 1.0, 0.0]]))
    assert loss(preds, targets).item() == pytest.approx(2.0)


def test_emd_grows_with_the_distance_the_mass_has_to_move():
    loss = EMDLoss("emd")
    near = loss(spike(5), spike(6)).item()
    far = loss(spike(5), spike(15)).item()
    assert 0 < near < far


def test_emd_is_zero_for_identical_spectra_and_averages_over_the_batch():
    loss = EMDLoss("emd")
    spectra = torch.rand(4, 10)
    assert loss(spectra, spectra).item() == 0.0

    preds = torch.cat([spike(0, 5), spike(0, 5)])
    targets = torch.cat([spike(0, 5), spike(4, 5)])
    assert loss(preds, targets).item() == pytest.approx(4.0 / 2)


def test_emd_rejects_unsupported_reductions():
    with pytest.raises(ValueError, match="Unsupported reduction"):
        EMDLoss("emd")(torch.ones(1, 3), torch.ones(1, 3), reduction="sum")


###############################################################################
##################################### WCC #####################################
###############################################################################


def test_wcc_is_zero_for_identical_spectra():
    spectra = torch.rand(3, 21) + 0.5
    assert WCCLoss("wcc", gaussian_hwhm=3)(spectra, spectra).item() == pytest.approx(0.0, abs=1e-5)


def test_wcc_ignores_the_overall_amplitude_of_the_prediction():
    targets = torch.rand(3, 21) + 0.5
    assert WCCLoss("wcc", gaussian_hwhm=3)(2.5 * targets, targets).item() == pytest.approx(0.0, abs=1e-5)


def test_wcc_penalises_peak_shifts_more_the_further_they_are():
    loss = WCCLoss("wcc", gaussian_hwhm=2)
    target = spike(10)

    shifts = [loss(spike(10 + shift), target).item() for shift in (0, 1, 3, 6)]

    assert shifts == sorted(shifts)
    assert shifts[0] == pytest.approx(0.0, abs=1e-5)
    assert shifts[-1] > shifts[1]


def test_a_wider_window_is_more_tolerant_to_shifts():
    narrow = WCCLoss("wcc", gaussian_hwhm=1)(spike(12), spike(10)).item()
    wide = WCCLoss("wcc", gaussian_hwhm=6)(spike(12), spike(10)).item()
    assert wide < narrow


@pytest.mark.parametrize("hwhm", [0, -3])
def test_wcc_requires_a_positive_window(hwhm):
    with pytest.raises(ValueError, match="gaussian_hwhm must be positive"):
        WCCLoss("wcc", gaussian_hwhm=hwhm)
