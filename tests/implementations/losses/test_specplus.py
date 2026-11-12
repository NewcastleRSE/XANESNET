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

"""Unit tests for the ``specplus`` loss (blurred MSE + detail + gradient terms)."""

import pytest
import torch

from xanesnet.losses import SpectralLossPlus

TARGETS = torch.sin(torch.linspace(0, 6, 40)).repeat(3, 1) * torch.tensor([[1.0], [2.0], [3.0]])


def make(**overrides) -> SpectralLossPlus:
    kwargs = {
        "blur_sigma_bins": 2.0,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "huber_delta": 0.01,
        "kappa_peak": 0.15,
    }
    kwargs.update(overrides)
    return SpectralLossPlus("specplus", **kwargs)


def test_identical_spectra_have_zero_loss():
    assert make()(TARGETS, TARGETS).item() == pytest.approx(0.0, abs=1e-7)


def test_loss_grows_with_the_error():
    loss = make()
    small = loss(TARGETS + 0.01, TARGETS).item()
    large = loss(TARGETS + 0.5, TARGETS).item()
    assert 0 < small < large


def test_mean_reduction_averages_the_loss_map():
    loss, preds = make(), TARGETS + 0.1 * torch.randn(3, 40, generator=torch.Generator().manual_seed(0))

    loss_map = loss(preds, TARGETS, reduction="none")

    assert loss_map.shape == TARGETS.shape
    torch.testing.assert_close(loss(preds, TARGETS), loss_map.mean())


def test_unsupported_reduction_is_rejected():
    with pytest.raises(ValueError, match="Unsupported reduction"):
        make()(TARGETS, TARGETS, reduction="sum")


@pytest.mark.parametrize("term", ["alpha", "beta", "gamma"])
def test_each_term_is_scaled_by_its_own_weight_and_switched_off_at_zero(term):
    preds = TARGETS + torch.randn(3, 40, generator=torch.Generator().manual_seed(1)) * 0.2
    only = {name: 0.0 for name in ("alpha", "beta", "gamma")}

    single = make(**{**only, term: 1.0})(preds, TARGETS)
    doubled = make(**{**only, term: 2.0})(preds, TARGETS)

    assert single.item() > 0
    assert doubled.item() == pytest.approx(2 * single.item(), rel=1e-5)
    assert make(**only)(preds, TARGETS).item() == 0.0


def test_total_loss_is_the_sum_of_the_three_terms():
    preds = TARGETS + torch.randn(3, 40, generator=torch.Generator().manual_seed(2)) * 0.2
    parts = [
        make(alpha=a, beta=b, gamma=g)(preds, TARGETS).item() for a, b, g in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    assert make()(preds, TARGETS).item() == pytest.approx(sum(parts), rel=1e-5)


def test_the_blur_preserves_constant_signals_and_returns_single_bins_unchanged():
    loss = make()
    constant = torch.full((2, 30), 3.0)
    torch.testing.assert_close(loss.gaussian_blur1d(constant, 2.0), constant, atol=1e-5, rtol=1e-5)

    single_bin = torch.tensor([[1.0], [2.0]])
    torch.testing.assert_close(loss.gaussian_blur1d(single_bin, 2.0), single_bin)


def test_the_blur_smooths_a_spike():
    spike = torch.zeros(1, 31)
    spike[0, 15] = 1.0

    blurred = make().gaussian_blur1d(spike, 2.0)

    assert blurred.max() < 1.0
    assert blurred.sum().item() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("kernel", [0, 4, -3])
def test_the_blur_kernel_size_must_be_a_positive_odd_integer(kernel):
    with pytest.raises(ValueError, match="positive odd integer"):
        make().gaussian_blur1d(TARGETS, 2.0, k=kernel)


def test_huber_loss_is_quadratic_below_delta_and_linear_above():
    loss = make()
    delta = 0.5
    small, large = torch.tensor([0.2]), torch.tensor([2.0])
    zero = torch.zeros(1)

    assert loss.huber_loss(small, zero, delta=delta).item() == pytest.approx(0.5 * 0.2**2)
    assert loss.huber_loss(large, zero, delta=delta).item() == pytest.approx(0.5 * delta**2 + delta * (2.0 - delta))
    with pytest.raises(ValueError, match="Unsupported reduction"):
        loss.huber_loss(small, zero, reduction="median")


def test_peak_weights_are_at_least_one_and_constant_for_gradient_flow():
    weights = make().peak_weighting(TARGETS.clone().requires_grad_(True))

    assert (weights >= 1.0).all()
    assert weights.requires_grad is False
