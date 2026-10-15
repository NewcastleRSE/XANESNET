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

"""Abstract base class for all XANESNET descriptors."""

from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor


class Descriptor(ABC):
    """Abstract base class for all XANESNET descriptors.

    Args:
        descriptor_type: Identifier string for the concrete descriptor type.
    """

    def __init__(
        self,
        descriptor_type: str,
    ) -> None:
        """Initialize ``Descriptor``."""
        self.descriptor_type = descriptor_type

    def transform_pmg(
        self,
        pmg_structure: Structure | Molecule,
        site_index: list[int] | int | None = 0,
    ) -> np.ndarray:
        """Convert a pymatgen structure to ASE and compute the descriptor.

        Args:
            pmg_structure: Pymatgen ``Structure`` or ``Molecule`` for the atomic system.
            site_index: Site index, list of site indices, or ``None`` for all sites.
                Defaults to ``0``.

        Returns:
            Descriptor feature array. Shape depends on the concrete descriptor.
        """
        ase_structure = AseAtomsAdaptor.get_atoms(pmg_structure)
        assert isinstance(ase_structure, Atoms), "Failed to convert pymatgen structure to ASE Atoms object."
        return self.transform(ase_structure, site_index=site_index)

    @abstractmethod
    def transform(
        self,
        system: Atoms,
        site_index: int | list[int] | None = 0,
    ) -> np.ndarray:
        """Compute the descriptor for one or more sites of an ASE ``Atoms`` object.

        Args:
            system: The atomic system.
            site_index: Site index, list of site indices, or ``None`` for all sites.
                Defaults to ``0``.

        Returns:
            Descriptor feature array. Shape depends on the concrete descriptor.
        """
        ...
