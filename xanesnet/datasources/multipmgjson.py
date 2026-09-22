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
#   ...

"""Datasource for multiple pymatgen JSON files stored across subdirectories."""

import logging
from collections.abc import Iterator
from pathlib import Path

from pymatgen.core import Molecule, Structure

from xanesnet.utils.exceptions import ResourceError
from xanesnet.utils.filesystem import list_filestems, list_subdir_stems

from .base import DataSource
from .pmgjson import PMGJSONSource
from .registry import DataSourceRegistry


@DataSourceRegistry.register("multipmgjson")
class MultiPMGJSONSource(DataSource):
    """Datasource for pymatgen JSON files across multiple subdirectories.

    Expects a root directory containing subdirectories, each holding one or
    more ``.json`` files. Each JSON file must contain a single serialised
    pymatgen ``Structure`` or ``Molecule`` entry, identified by the
    ``@class`` key.

    Args:
        datasource_type: Identifier string for this datasource type.
        root_path: Path to the root directory containing the subdirectories.
        spectrum_key: Site-property key under which the spectrum is stored
            in the pymatgen objects. The datasource remaps it to
            ``"spectrum"`` so that downstream code always sees a uniform
            key.
    """

    def __init__(
        self,
        datasource_type: str,
        root_path: str,
        spectrum_key: str,
    ) -> None:
        """Initialize ``MultiPMGJSONSource``."""
        super().__init__(datasource_type)

        self.root_path = root_path
        self.spectrum_key = spectrum_key

        self.sample_ids: dict[str, list[str]] = self._get_file_dictionary()
        self._subdir_ids: dict[str, int] = {subdir: subdir_id for subdir_id, subdir in enumerate(self.sample_ids)}
        self._flat_index: list[tuple[str, str]] = [
            (subdir, file) for subdir, files in self.sample_ids.items() for file in files
        ]

    def __iter__(self) -> Iterator[Molecule | Structure]:
        """Iterate over all entries in the datasource.

        Returns:
            Iterator over loaded pymatgen entries.
        """
        for i in range(len(self._flat_index)):
            yield self[i]

    def __len__(self) -> int:
        """Return the total number of entries across all subdirectories.

        Returns:
            Number of JSON files available for loading.
        """
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> Molecule | Structure:
        """Return the structure or molecule at the given flat index.

        Args:
            idx: Zero-based flat index across all subdirectories.

        Returns:
            The deserialised pymatgen ``Molecule`` or ``Structure`` at
            position ``idx``, with ``sample_id``, ``subdir_name``, and
            ``subdir_id`` stored in ``properties``. When ``spectrum_key`` is
            not ``"spectrum"``, the source property is available under
            ``"spectrum"`` and the original property is removed.
        """
        subdir, file = self._flat_index[idx]
        json_file = Path(self.root_path) / subdir / f"{file}.json"

        structure = PMGJSONSource.load_json(json_file)
        structure.properties["sample_id"] = file
        structure.properties["subdir_name"] = subdir
        structure.properties["subdir_id"] = self._subdir_ids[subdir]

        if self.spectrum_key != "spectrum" and self.spectrum_key in structure.site_properties:
            structure.add_site_property("spectrum", structure.site_properties[self.spectrum_key])
            structure.remove_site_property(self.spectrum_key)

        return structure

    def _get_file_dictionary(self) -> dict[str, list[str]]:
        """Build a mapping from subdirectory names to their JSON sample identifiers.

        Only files ending in ``.json`` are considered. Unrelated files are
        ignored.

        Returns:
            Mapping from subdirectory name to sorted list of sample identifiers.

        Raises:
            ResourceError: If the root path does not exist, no subdirectories
                are found, or no JSON files are found across all
                subdirectories.
        """
        root_path = Path(self.root_path)
        if not root_path.is_dir():
            raise ResourceError(f"Root path does not exist: {root_path}")

        subdirectories = list_subdir_stems(root_path)
        if not subdirectories:
            raise ResourceError(f"No subdirectories found in root path: {self.root_path}")

        files_dict: dict[str, list[str]] = {}
        for subdir in subdirectories:
            json_dir = root_path / subdir
            sample_ids = sorted(list_filestems(json_dir, suffixes=".json"))

            if not sample_ids:
                logging.warning(f"No JSON files found in subdirectory: {subdir}")
                continue

            files_dict[subdir] = sample_ids

        if not files_dict:
            raise ResourceError(f"No JSON files found in any subdirectories of root path: {self.root_path}")

        return files_dict
