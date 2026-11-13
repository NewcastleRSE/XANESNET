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

"""Tests for the registered ``direct`` descriptor (features read from files)."""

import numpy as np
import pytest
from ase import Atoms

from xanesnet.descriptors import DescriptorRegistry

FEATURES = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


def atoms(sample_id: str = "s1") -> Atoms:
    system = Atoms("H3", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    system.info["sample_id"] = sample_id
    return system


@pytest.fixture
def source_dir(tmp_path):
    np.savetxt(tmp_path / "s1.txt", FEATURES)
    np.savetxt(tmp_path / "flat.txt", np.array([1.0, 2.0, 3.0]))
    return tmp_path


def direct(source_dir, preload: bool = False):
    return DescriptorRegistry.create("direct", descriptor_type="direct", source_dir=str(source_dir), preload=preload)


###############################################################################
#################################### DIRECT ###################################
###############################################################################


@pytest.mark.parametrize(
    ("site_index", "expected"),
    [(0, FEATURES[[0]]), (2, FEATURES[[2]]), ([2, 0], FEATURES[[2, 0]]), (None, FEATURES)],
)
def test_features_are_selected_per_site(source_dir, site_index, expected):
    np.testing.assert_array_equal(direct(source_dir).transform(atoms(), site_index=site_index), expected)


def test_the_file_is_chosen_by_the_sample_id(source_dir):
    np.savetxt(source_dir / "other.txt", FEATURES * 10)

    np.testing.assert_array_equal(direct(source_dir).transform(atoms("other"), site_index=None), FEATURES * 10)


def test_single_row_files_are_returned_as_two_dimensional_features(source_dir):
    result = direct(source_dir).transform(atoms("flat"), site_index=None)
    assert result.shape == (1, 3)


def test_preloaded_features_are_served_from_memory(source_dir):
    descriptor = direct(source_dir, preload=True)
    (source_dir / "s1.txt").unlink()

    np.testing.assert_array_equal(descriptor.transform(atoms(), site_index=None), FEATURES)


def test_features_without_preloading_are_read_on_demand(source_dir):
    descriptor = direct(source_dir, preload=False)
    (source_dir / "s1.txt").unlink()

    with pytest.raises(OSError):
        descriptor.transform(atoms())
