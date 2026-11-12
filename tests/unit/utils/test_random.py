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

"""Unit tests for ``set_global_seed``."""

import random

import numpy as np
import pytest
import torch

from xanesnet.utils.random import set_global_seed


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_explicit_seed_is_returned_and_makes_all_generators_reproducible():
    assert set_global_seed(123) == 123
    first = _draw()
    set_global_seed(123)
    assert _draw() == first


def test_different_seeds_give_different_streams():
    set_global_seed(1)
    first = _draw()
    set_global_seed(2)
    assert _draw() != first


def test_none_draws_a_valid_random_seed_and_returns_it():
    seed = set_global_seed(None)
    assert isinstance(seed, int)
    assert 0 <= seed <= 2**32 - 1

    # The returned seed must reproduce the stream that was just seeded.
    first = _draw()
    set_global_seed(seed)
    assert _draw() == first


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_boundary_seeds_are_accepted(seed):
    assert set_global_seed(seed) == seed


def test_numpy_integers_are_accepted_and_converted_to_int():
    seed = set_global_seed(np.int64(5))
    assert seed == 5
    assert type(seed) is int


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_out_of_range_seed_raises_value_error(seed):
    with pytest.raises(ValueError, match="between 0 and"):
        set_global_seed(seed)


@pytest.mark.parametrize("seed", ["1", 1.5, [1]])
def test_non_integer_seed_raises_type_error(seed):
    with pytest.raises(TypeError, match="integer or None"):
        set_global_seed(seed)
