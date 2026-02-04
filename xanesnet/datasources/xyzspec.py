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

import os
from pathlib import Path
from typing import Iterator

import numpy as np
from pymatgen.core import Molecule

from xanesnet.utils import ResourceError, list_filestems

from .base import DataSource
from .registry import DataSourceRegistry

###############################################################################
#################################### CLASS ####################################
###############################################################################


@DataSourceRegistry.register("xyzspec")
class XYZSpecSource(DataSource):
    def __init__(
        self,
        datasource_type: str,
        xyz_path: str,
        xanes_path: str,
    ) -> None:
        super().__init__(datasource_type)

        # TODO Currently the paths cannot be None!
        # TODO We might change this in the future to allow only one of them.
        # TODO This can be beneficial for prediction datasets without spectra.
        self.xyz_path = xyz_path
        self.xanes_path = xanes_path

        self.file_names: list[str] = self._get_file_list()

    def __iter__(self) -> Iterator[Molecule]:
        for file in self.file_names:
            xyz_file = os.path.join(self.xyz_path, f"{file}.xyz")
            xanes_file = os.path.join(self.xanes_path, f"{file}.txt")

            molecule = self.load_xyz(xyz_file)
            energies, intensities = self.load_xanes(xanes_file)
            spectra_list: list[dict[str, np.ndarray] | None] = [None for _ in molecule.sites]
            spectra_list[0] = {
                "energies": energies,
                "intensities": intensities,
            }
            molecule.add_site_property("XANES", spectra_list)
            molecule.properties["file_name"] = file
            yield molecule

    def __len__(self) -> int:
        return len(self.file_names)

    def _get_file_list(self) -> list[str]:
        """
        Get the list of valid file stems based on the
        xyz_path and/or xanes_path. If both are given, only common stems are kept.
        """
        xyz_path = Path(self.xyz_path)
        xanes_path = Path(self.xanes_path)

        xyz_stems = set(list_filestems(xyz_path))
        xanes_stems = set(list_filestems(xanes_path))
        file_names = sorted(list(xyz_stems & xanes_stems))

        if not file_names:
            raise ResourceError("No matching files found in the provided paths.")

        return file_names

    @staticmethod
    def load_xanes(file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load XANES spectrum from a file.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        # pop the FDMNES header block
        for _ in range(2):
            lines.pop(0)

        xanes_block = [lines.pop(0).split() for _ in range(len(lines))]
        energies = np.array([line[0] for line in xanes_block], dtype="float32")
        intensities = np.array([line[1] for line in xanes_block], dtype="float32")

        return energies, intensities

    @staticmethod
    def load_xyz(file_path: str) -> Molecule:
        """
        Load XYZ coordinates from a file.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        n_atoms = int(lines.pop(0).strip())
        comment = lines.pop(0).strip()

        atoms_block = [lines.pop(0).split() for _ in range(n_atoms)]
        elements = [line[0] for line in atoms_block]
        coords = np.array([line[1:] for line in atoms_block], dtype="float32")

        molecule = Molecule(elements, coords, properties={"comment": comment})

        return molecule
