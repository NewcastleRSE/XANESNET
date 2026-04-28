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

"""Datasource for multiple paired XYZ coordinate files and XANES spectra across subdirectories."""

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pymatgen.core import Molecule

from xanesnet.utils.exceptions import ResourceError
from xanesnet.utils.filesystem import list_filestems, list_subdir_stems

from .base import DataSource
from .registry import DataSourceRegistry


@DataSourceRegistry.register("multixyzspec")
class MultiXYZSpecSource(DataSource):
    """Datasource for multiple paired XYZ coordinate files and XANES spectra.

    Expects a root directory containing subdirectories, each with an ``xyz/``
    and a ``spectra/`` subdirectory. Within each subdirectory, ``.xyz`` and
    ``.txt`` file stems must match one-to-one.

    Args:
        datasource_type: Identifier string for this datasource type.
        root_path: Path to the root directory containing the subdirectories.
    """

    def __init__(
        self,
        datasource_type: str,
        root_path: str,
    ) -> None:
        """Initialize ``MultiXYZSpecSource``."""
        super().__init__(datasource_type)

        self.root_path = root_path

        self.file_names: dict[str, list[str]] = self._get_file_dictionary()
        self._flat_index: list[tuple[str, str]] = [
            (subdir, file) for subdir, files in self.file_names.items() for file in files
        ]

    def __iter__(self) -> Iterator[Molecule]:
        """Iterate over all molecule entries in the datasource."""
        for i in range(len(self._flat_index)):
            yield self[i]

    def __len__(self) -> int:
        """Return the total number of entries across all subdirectories."""
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> Molecule:
        """Return the molecule at the given flat index.

        Args:
            idx: Zero-based flat index across all subdirectories.

        Returns:
            A ``Molecule`` with ``XANES`` site property and ``file_name``
            stored in ``properties``.
        """
        subdir, file = self._flat_index[idx]
        xyz_file = Path(self.root_path) / subdir / "xyz" / f"{file}.xyz"
        spectra_file = Path(self.root_path) / subdir / "spectra" / f"{file}.txt"

        molecule = self.load_xyz(xyz_file)
        energies, intensities = self.load_xanes(spectra_file)
        spectra_list: list[dict[str, np.ndarray] | None] = [None for _ in molecule.sites]
        spectra_list[0] = {
            "energies": energies,
            "intensities": intensities,
        }
        molecule.add_site_property("XANES", spectra_list)
        molecule.properties["file_name"] = file
        return molecule

    def _get_file_dictionary(self) -> dict[str, list[str]]:
        """Build a mapping from subdirectory names to their matched file stems.

        Only stems that have both a ``.xyz`` file in ``xyz/`` and a ``.txt``
        file in ``spectra/`` are included. Unrelated files are ignored.

        Returns:
            Mapping from subdirectory name to sorted list of matched file stems.

        Raises:
            ResourceError: If the root path does not exist, no subdirectories
                are found, a subdirectory is missing the expected ``xyz/`` or
                ``spectra/`` sub-folders, or no valid file pairs are found
                across all subdirectories.
        """
        root_path = Path(self.root_path)
        if not root_path.is_dir():
            raise ResourceError(f"Root path does not exist: {root_path}")

        subdirectories = list_subdir_stems(root_path)
        if not subdirectories:
            raise ResourceError(f"No subdirectories found in root path: {self.root_path}")

        files_dict: dict[str, list[str]] = {}
        for subdir in subdirectories:
            xyz_dir = root_path / subdir / "xyz"
            spectra_dir = root_path / subdir / "spectra"

            if not xyz_dir.is_dir() or not spectra_dir.is_dir():
                raise ResourceError(f"Subdirectory {subdir} must contain 'xyz' and 'spectra' subdirectories.")

            xyz_files = set(list_filestems(xyz_dir, suffixes=".xyz"))
            spectra_files = set(list_filestems(spectra_dir, suffixes=".txt"))
            file_names = sorted(list(xyz_files & spectra_files))

            if not file_names:
                logging.warning(f"No matching .xyz and .txt files found in subdirectory: {subdir}")
                continue

            files_dict[subdir] = file_names

        if not files_dict:
            raise ResourceError(f"No valid file pairs found in any subdirectories of root path: {self.root_path}")

        return files_dict

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
