# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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
                Defaults to ``0`` (the absorber site).

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
                Defaults to ``0`` (the absorber site).

        Returns:
            Descriptor feature array. Shape depends on the concrete descriptor.
        """
        ...
