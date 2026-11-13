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

"""Unit tests for the ``msssim`` loss (multi-scale structural similarity for 1D spectra)."""

from typing import Any

import pytest
import torch

from xanesnet.losses import MultiScale_SSIM

TARGETS = (torch.sin(torch.linspace(0, 8, 64)).repeat(3, 1) + 1.5) * torch.tensor([[1.0], [1.5], [2.0]])


def make(**overrides) -> MultiScale_SSIM:
    kwargs: dict[str, Any] = {
        "fractions": [0.05, 0.1, 0.2],
        "data_range": 4.0,
        "K": (0.01, 0.03),
        "use_weighted_sum": True,
        "weights": None,
    }
    kwargs.update(overrides)
    return MultiScale_SSIM("msssim", **kwargs)


def test_identical_spectra_have_zero_loss():
    assert make()(TARGETS, TARGETS).item() == pytest.approx(0.0, abs=1e-5)


def test_loss_grows_with_the_distortion():
    loss = make()
    noise = torch.randn(3, 64, generator=torch.Generator().manual_seed(0))

    small = loss(TARGETS + 0.05 * noise, TARGETS).item()
    large = loss(TARGETS + 0.5 * noise, TARGETS).item()

    assert 0 < small < large


def test_none_reduction_returns_one_value_per_spectral_point():
    loss = make()
    preds = TARGETS + 0.2

    loss_map = loss(preds, TARGETS, reduction="none")

    assert loss_map.shape == TARGETS.shape
    torch.testing.assert_close(loss(preds, TARGETS), loss_map.mean())


def test_unsupported_reduction_is_rejected():
    with pytest.raises(ValueError, match="Unsupported reduction"):
        make()(TARGETS, TARGETS, reduction="sum")


def test_with_a_single_scale_the_weighted_sum_equals_the_product():
    preds = TARGETS + 0.3
    weighted = make(fractions=[0.1], use_weighted_sum=True)(preds, TARGETS)
    product = make(fractions=[0.1], use_weighted_sum=False)(preds, TARGETS)
    torch.testing.assert_close(weighted, product)


def test_explicit_scale_weights_change_the_result_and_must_match_the_number_of_scales():
    preds = TARGETS + 0.3 * torch.randn(3, 64, generator=torch.Generator().manual_seed(3))

    uniform = make()(preds, TARGETS).item()
    skewed = make(weights=[1.0, 0.0, 0.0])(preds, TARGETS).item()

    assert uniform != pytest.approx(skewed)
    with pytest.raises(ValueError, match="must match number of scales"):
        make(weights=[1.0, 1.0])


def test_without_a_fixed_data_range_it_is_learned_from_the_targets_and_checkpointed():
    adaptive = make(data_range=None)
    wide_targets = TARGETS * 10
    first = adaptive(wide_targets + 1.0, wide_targets).item()

    restored = make(data_range=None)
    restored.load_state_dict(adaptive.state_dict())

    assert restored(wide_targets + 1.0, wide_targets).item() == pytest.approx(first)
    # the stabilising constants grew with the observed range and were restored with it
    assert restored.get_extra_state() == adaptive.get_extra_state()
    assert adaptive.get_extra_state()[3] > make(data_range=None).get_extra_state()[3]
