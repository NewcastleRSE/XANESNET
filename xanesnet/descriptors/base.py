"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

###############################################################################
################################## CLASSES ####################################
###############################################################################


class Descriptor(ABC):
    """Abstract base class for XANESNET descriptors."""

    def __init__(
        self,
        descriptor_type: str,
    ) -> None:
        self.descriptor_type = descriptor_type

    def transform_pmg(
        self,
        pmg_structure: Structure | Molecule,
        site_index: list[int] | int | None = 0,
    ) -> np.ndarray:
        """
        Args:
            pmg_structure (Structure | Molecule): Pymatgen Structure or Molecule
                representing the atomic system.
            site_index (int | list[int] | None): Index or list of indices of the sites to compute the descriptor for.
                If None, computes descriptors for all sites. Defaults to 0 (absorber).

        Returns:
            np.ndarray: A fingerprint feature vector for the molecular system.
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
        """
        Args:
            system (Atoms): A molecular system.
            site_index (int | list[int] | None): Index or list of indices of the sites to compute the descriptor for.
                If None, computes descriptors for all sites. Defaults to 0 (absorber).

        Returns:
            np.ndarray: A fingerprint feature vector for the molecular system.
        """
        ...
