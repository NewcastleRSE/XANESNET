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

"""Unit tests for the ``rdc`` and ``wacsf`` descriptors (hand-written radial/angular symmetry functions)."""

import numpy as np
import pytest
from ase import Atoms

from xanesnet.descriptors import DescriptorRegistry


def rdc(**overrides):
    kwargs = {"r_min": 0.0, "r_max": 4.0, "dr": 1.0, "alpha": 2.0, "use_charge": False, "use_spin": False}
    kwargs.update(overrides)
    return DescriptorRegistry.create("rdc", descriptor_type="rdc", **kwargs)


def wacsf(**overrides):
    kwargs = {
        "r_min": 0.0,
        "r_max": 4.0,
        "n_g2": 0,
        "n_g4": 0,
        "l": None,
        "z": None,
        "g2_parameterisation": "shifted",
        "g4_parameterisation": "shifted",
        "use_charge": False,
        "use_spin": False,
    }
    kwargs.update(overrides)
    return DescriptorRegistry.create("wacsf", descriptor_type="wacsf", **kwargs)


def dimer(distance: float = 1.5, symbols: str = "OH") -> Atoms:
    return Atoms(symbols, positions=[[0, 0, 0], [0, 0, distance]])


def water() -> Atoms:
    return Atoms("OHH", positions=[[0.0, 0.0, 0.1], [0.0, 0.76, -0.47], [0.1, -0.76, -0.47]])


def rotate(system: Atoms) -> Atoms:
    rotated = system.copy()
    rotated.rotate(37, "xyz", rotate_cell=False)
    rotated.translate([1.0, -2.0, 0.5])
    return rotated


def cosine_cutoff(r: float, r_max: float) -> float:
    return (np.cos(np.pi * r / r_max) + 1) / 2


###############################################################################
##################################### RDC #####################################
###############################################################################


def test_rdc_is_a_sum_of_gaussians_weighted_by_the_atomic_number_product():
    descriptor = rdc()
    r_aux = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    result = descriptor.transform(dimer(1.5), site_index=0)

    expected = 8 * 1 * np.exp(-2.0 * (1.5 - r_aux) ** 2)
    assert result.shape == (1, 5)
    np.testing.assert_allclose(result[0], expected)


def test_rdc_grid_follows_the_configured_range_and_resolution():
    descriptor = rdc(r_min=1.0, r_max=3.0, dr=0.5)
    np.testing.assert_allclose(descriptor.r_aux, [1.0, 1.5, 2.0, 2.5, 3.0])
    assert descriptor.transform(dimer(1.5)).shape == (1, 5)


def test_rdc_rows_follow_the_requested_sites_and_none_means_all_sites():
    descriptor = rdc()
    system = dimer(1.5)

    both = descriptor.transform(system, site_index=None)
    swapped = descriptor.transform(system, site_index=[1, 0])

    assert both.shape == (2, 5)
    np.testing.assert_allclose(swapped, both[[1, 0]])


def test_rdc_ignores_atoms_beyond_the_cutoff():
    near = rdc(r_max=4.0).transform(dimer(1.5) + Atoms("H", positions=[[0, 0, 40.0]]), site_index=0)
    alone = rdc(r_max=4.0).transform(dimer(1.5), site_index=0)

    np.testing.assert_allclose(near, alone)


def test_rdc_optionally_appends_spin_then_charge():
    system = dimer()
    system.info.update(S=2, q=-1)

    plain = rdc().transform(system)
    extended = rdc(use_spin=True, use_charge=True).transform(system)

    assert extended.shape[1] == plain.shape[1] + 2
    assert extended[0, -2:].tolist() == [2.0, -1.0]


def test_rdc_cannot_describe_an_atom_without_neighbours():
    with pytest.raises(RuntimeError, match="Too few atoms"):
        rdc().transform(dimer(distance=30.0), site_index=0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"dr": 0.0}, "dr must be positive"), ({"r_min": 5.0, "r_max": 1.0}, "r_max")],
)
def test_rdc_validates_its_grid(overrides, message):
    with pytest.raises(ValueError, match=message):
        rdc(**overrides)


###############################################################################
#################################### WACSF ####################################
###############################################################################


@pytest.mark.parametrize(
    ("n_g2", "n_g4", "width"),
    [(0, 0, 1), (3, 0, 4), (0, 4, 5), (2, 6, 9)],
)
def test_wacsf_has_one_radial_sum_then_g2_then_g4_values_per_site(n_g2, n_g4, width):
    result = wacsf(n_g2=n_g2, n_g4=n_g4, l=[1.0, -1.0], z=[1.0, 2.0] if n_g4 % 4 == 0 else [1.0]).transform(water())

    assert result.shape == (1, width)


def test_g1_is_the_sum_of_cosine_cutoffs_over_the_neighbours():
    result = wacsf().transform(dimer(1.5), site_index=0)
    np.testing.assert_allclose(result, [[cosine_cutoff(1.5, 4.0)]])


def test_g1_vanishes_at_the_cutoff_radius():
    np.testing.assert_allclose(wacsf(r_max=4.0).transform(dimer(4.0), site_index=0), [[0.0]], atol=1e-12)


def test_g2_weights_neighbours_by_a_tenth_of_their_atomic_number():
    descriptor = wacsf(n_g2=1)
    r = 1.5

    result = descriptor.transform(dimer(r, "OH"), site_index=0)

    # single 'shifted' function: centre (r_min + r_max) / 2 and width 1 / (2 (r_max - r_min)^2)
    h, m = 1 / (2 * 4.0**2), 2.0
    expected_g2 = 0.1 * 1 * np.exp(-h * (r - m) ** 2) * cosine_cutoff(r, 4.0)
    np.testing.assert_allclose(result[0, 1], expected_g2)


def test_atoms_without_neighbours_give_zeros_followed_by_spin_and_charge():
    system = dimer(distance=30.0)
    system.info.update(S=1, q=2)

    result = wacsf(n_g2=2, n_g4=2, use_spin=True, use_charge=True).transform(system, site_index=0)

    assert result.tolist() == [[0.0] * 5 + [1.0, 2.0]]


def test_spin_and_charge_are_appended_in_that_order():
    system = water()
    system.info.update(S=1, q=2)

    result = wacsf(n_g2=2, use_spin=True, use_charge=True).transform(system, site_index=0)

    assert result.shape == (1, 5)
    assert result[0, -2:].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("descriptor", [rdc(), wacsf(n_g2=3, n_g4=2, l=[1.0, -1.0], z=[1.0])])
def test_descriptors_do_not_change_under_rotation_and_translation(descriptor):
    original = descriptor.transform(water(), site_index=None)
    moved = descriptor.transform(rotate(water()), site_index=None)

    np.testing.assert_allclose(moved, original, atol=1e-8)


def test_g4_sums_over_pairs_of_neighbours_and_distinguishes_angles():
    descriptor = wacsf(n_g4=2, l=[1.0, -1.0], z=[1.0])
    right_angle = Atoms("OHH", positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    straight = Atoms("OHH", positions=[[0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]])

    right = descriptor.transform(right_angle, site_index=0)[0, 1:]
    line = descriptor.transform(straight, site_index=0)[0, 1:]

    # lambda = +1 favours small angles, lambda = -1 favours large ones
    assert right[0] > line[0]
    assert line[1] > right[1]


def test_g4_needs_a_neighbour_count_divisible_by_the_lambda_zeta_grid():
    with pytest.raises(ValueError, match="Can't generate 3 G4 symmetry functions"):
        wacsf(n_g4=3, l=[1.0, -1.0], z=[1.0])


@pytest.mark.parametrize("overrides", [{"n_g2": -1}, {"n_g4": -1}])
def test_negative_counts_are_rejected(overrides):
    with pytest.raises(ValueError, match="non-negative"):
        wacsf(**overrides)


@pytest.mark.parametrize("which", ["g2_parameterisation", "g4_parameterisation"])
def test_unknown_parameterisations_are_rejected(which):
    count = {"g2_parameterisation": {"n_g2": 2}, "g4_parameterisation": {"n_g4": 2}}[which]

    with pytest.raises(ValueError, match="parameterisation must be"):
        wacsf(**count, **{which: "other"})


@pytest.mark.parametrize("parameterisation", ["shifted", "centred"])
def test_both_parameterisations_produce_finite_features(parameterisation):
    result = wacsf(n_g2=3, g2_parameterisation=parameterisation).transform(water(), site_index=None)
    assert np.isfinite(result).all()
