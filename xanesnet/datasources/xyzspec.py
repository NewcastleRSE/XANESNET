# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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

"""Datasource for paired XYZ coordinate files and spectra."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pymatgen.core import Molecule

from xanesnet.utils.exceptions import ResourceError
from xanesnet.utils.filesystem import list_filestems

from .base import DataSource
from .registry import DataSourceRegistry

# TODO Document somewhere how exactly the spectrum files are expected to look like.


@DataSourceRegistry.register("xyzspec")
class XYZSpecSource(DataSource):
    """Datasource for paired XYZ coordinate files and spectra files.

    Expects two directories: one with ``.xyz`` files and one with ``.txt``
    spectra files. File stems (names without extension) must match between the
    two directories.

    Args:
        datasource_type: Identifier string for this datasource type.
        xyz_path: Path to the directory containing the ``.xyz`` files.
        spectra_path: Path to the directory containing the ``.txt`` spectra files.
    """

    def __init__(
        self,
        datasource_type: str,
        xyz_path: str,
        spectra_path: str,
    ) -> None:
        """Initialize ``XYZSpecSource``."""
        super().__init__(datasource_type)

        # TODO Currently the paths cannot be None!
        # TODO We might change this in the future to allow only one of them.
        # TODO This can be beneficial for prediction datasets without spectra.
        self.xyz_path = xyz_path
        self.spectra_path = spectra_path

        self.sample_ids: list[str] = self._get_file_list()

    def __iter__(self) -> Iterator[Molecule]:
        """Iterate over all molecule entries in the datasource.

        Returns:
            Iterator over loaded molecule entries.
        """
        for i in range(len(self.sample_ids)):
            yield self[i]

    def __len__(self) -> int:
        """Return the total number of entries in the datasource.

        Returns:
            Number of matched XYZ/spectrum pairs.
        """
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Molecule:
        """Return the molecule at the given index.

        Args:
            idx: Zero-based index into the datasource.

        Returns:
            A ``Molecule`` with ``"spectrum"`` site property and ``sample_id``
            stored in ``properties``.
        """
        file = self.sample_ids[idx]
        xyz_file = Path(self.xyz_path) / f"{file}.xyz"
        spectra_file = Path(self.spectra_path) / f"{file}.txt"

        molecule = self.load_xyz(xyz_file)
        energies, intensities = self.load_spectrum(spectra_file)
        spectra_list: list[dict[str, np.ndarray] | None] = [None for _ in molecule.sites]
        spectra_list[0] = {
            "energies": energies,
            "intensities": intensities,
        }
        molecule.add_site_property("spectrum", spectra_list)
        molecule.properties["sample_id"] = file
        return molecule

    def _get_file_list(self) -> list[str]:
        """Build the sorted list of sample identifiers common to both ``xyz_path`` and ``spectra_path``.

        Only ``.xyz`` files from ``xyz_path`` and ``.txt`` files from
        ``spectra_path`` are considered. Unrelated files are ignored.

        Returns:
            Sorted list of matched sample identifiers.

        Raises:
            ResourceError: If either path is not a directory or no matching
                ``.xyz``/``.txt`` sample identifiers are found.
        """
        xyz_path = Path(self.xyz_path)
        spectra_path = Path(self.spectra_path)

        if not xyz_path.is_dir():
            raise ResourceError(f"XYZ directory does not exist: {xyz_path}")
        if not spectra_path.is_dir():
            raise ResourceError(f"Spectra directory does not exist: {spectra_path}")

        xyz_stems = set(list_filestems(xyz_path, suffixes=".xyz"))
        spectra_stems = set(list_filestems(spectra_path, suffixes=".txt"))
        sample_ids = sorted(list(xyz_stems & spectra_stems))

        if not sample_ids:
            raise ResourceError(f"No matching .xyz and .txt files found in: {xyz_path} and {spectra_path}")

        return sample_ids

    @staticmethod
    def load_spectrum(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load a spectrum from a two-column simulation output text file.

        Skips the required two-line header block at the top of the file.

        Args:
            file_path: Path to the ``.txt`` spectra file.

        Returns:
            Tuple of ``(energies, intensities)`` as float32 arrays.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Skip the required header block.
        for _ in range(2):
            lines.pop(0)

        spectrum_block = [lines.pop(0).split() for _ in range(len(lines))]
        energies = np.array([line[0] for line in spectrum_block], dtype="float32")
        intensities = np.array([line[1] for line in spectrum_block], dtype="float32")

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
