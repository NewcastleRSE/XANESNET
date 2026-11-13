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

"""Unit tests for the shared descriptor base class."""

import numpy as np
import pytest
from ase import Atoms
from pymatgen.core import Molecule

from xanesnet.descriptors import Descriptor


def test_pymatgen_objects_are_converted_to_ase_atoms_before_transforming():
    received = {}

    class Recorder(Descriptor):
        """Descriptor stub that captures the converted system and site index."""

        def transform(self, system, site_index=0):
            received.update(system=system, site_index=site_index)
            return np.zeros(1)

    molecule = Molecule(["O", "H"], [[0, 0, 0], [0, 0, 1]])

    Recorder("recorder").transform_pmg(molecule, site_index=[1])

    assert isinstance(received["system"], Atoms)
    assert received["system"].get_chemical_symbols() == ["O", "H"]
    assert received["site_index"] == [1]


def test_pymatgen_transform_defaults_to_the_first_site():
    class Recorder(Descriptor):
        """Descriptor stub that returns the selected site as one feature."""

        def transform(self, system, site_index=0):
            return np.array([site_index])

    assert Recorder("recorder").transform_pmg(Molecule(["H"], [[0, 0, 0]])).tolist() == [0]


def test_descriptor_is_abstract():
    with pytest.raises(TypeError):
        Descriptor("base")  # type: ignore[abstract]
