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

"""Datasource for paired XYZ coordinate files and XANES spectra."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pymatgen.core import Molecule

from xanesnet.utils.exceptions import ResourceError
from xanesnet.utils.filesystem import list_filestems

from .base import DataSource
from .registry import DataSourceRegistry


@DataSourceRegistry.register("xyzspec")
class XYZSpecSource(DataSource):
    """Datasource for paired XYZ coordinate files and XANES spectra.

    Expects two directories: one with ``.xyz`` files and one with ``.txt``
    spectra files. File stems (names without extension) must match between the
    two directories.

    Args:
        datasource_type: Identifier string for this datasource type.
        xyz_path: Path to the directory containing the ``.xyz`` files.
        xanes_path: Path to the directory containing the ``.txt`` spectra files.
    """

    def __init__(
        self,
        datasource_type: str,
        xyz_path: str,
        xanes_path: str,
    ) -> None:
        """Initialize ``XYZSpecSource``."""
        super().__init__(datasource_type)

        # TODO Currently the paths cannot be None!
        # TODO We might change this in the future to allow only one of them.
        # TODO This can be beneficial for prediction datasets without spectra.
        self.xyz_path = xyz_path
        self.xanes_path = xanes_path

        self.file_names: list[str] = self._get_file_list()

    def __iter__(self) -> Iterator[Molecule]:
        """Iterate over all molecule entries in the datasource."""
        for i in range(len(self.file_names)):
            yield self[i]

    def __len__(self) -> int:
        """Return the total number of entries in the datasource."""
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Molecule:
        """Return the molecule at the given index.

        Args:
            idx: Zero-based index into the datasource.

        Returns:
            A ``Molecule`` with ``XANES`` site property and ``file_name``
            stored in ``properties``.
        """
        file = self.file_names[idx]
        xyz_file = Path(self.xyz_path) / f"{file}.xyz"
        xanes_file = Path(self.xanes_path) / f"{file}.txt"

        molecule = self.load_xyz(xyz_file)
        energies, intensities = self.load_xanes(xanes_file)
        spectra_list: list[dict[str, np.ndarray] | None] = [None for _ in molecule.sites]
        spectra_list[0] = {
            "energies": energies,
            "intensities": intensities,
        }
        molecule.add_site_property("XANES", spectra_list)
        molecule.properties["file_name"] = file
        return molecule

    def _get_file_list(self) -> list[str]:
        """Build the sorted list of file stems common to both ``xyz_path`` and ``xanes_path``.

        Only ``.xyz`` files from ``xyz_path`` and ``.txt`` files from
        ``xanes_path`` are considered. Unrelated files are ignored.

        Returns:
            Sorted list of matched file stems.

        Raises:
            ResourceError: If either path is not a directory or no matching
                ``.xyz``/``.txt`` file stems are found.
        """
        xyz_path = Path(self.xyz_path)
        xanes_path = Path(self.xanes_path)

        if not xyz_path.is_dir():
            raise ResourceError(f"XYZ directory does not exist: {xyz_path}")
        if not xanes_path.is_dir():
            raise ResourceError(f"XANES directory does not exist: {xanes_path}")

        xyz_stems = set(list_filestems(xyz_path, suffixes=".xyz"))
        xanes_stems = set(list_filestems(xanes_path, suffixes=".txt"))
        file_names = sorted(list(xyz_stems & xanes_stems))

        if not file_names:
            raise ResourceError(f"No matching .xyz and .txt files found in: {xyz_path} and {xanes_path}")

        return file_names

    @staticmethod
    def load_xanes(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load a XANES spectrum from an FDMNES output text file.

        Skips the two-line FDMNES header block at the top of the file.

        Args:
            file_path: Path to the ``.txt`` spectra file.

        Returns:
            Tuple of ``(energies, intensities)`` as float32 arrays.
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
    def load_xyz(file_path: Path) -> Molecule:
        """Load an XYZ coordinate file into a pymatgen ``Molecule``.

        Args:
            file_path: Path to the ``.xyz`` file.

        Returns:
            A ``Molecule`` with the ``comment`` line stored in ``properties``.
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
