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

"""Datasource for pymatgen JSON files stored in a single directory."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from pymatgen.core import Molecule, Structure

from xanesnet.utils.exceptions import ResourceError
from xanesnet.utils.filesystem import list_filestems

from .base import DataSource
from .registry import DataSourceRegistry


@DataSourceRegistry.register("pmgjson")
class PMGJSONSource(DataSource):
    """Datasource for pymatgen JSON files in a single directory.

    Each JSON file must contain a single serialised pymatgen ``Structure`` or
    ``Molecule`` entry, identified by the ``@class`` key.

    Args:
        datasource_type: Identifier string for this datasource type.
        json_path: Path to the directory containing the ``.json`` files.
    """

    def __init__(
        self,
        datasource_type: str,
        json_path: str,
    ) -> None:
        """Initialize ``PMGJSONSource``."""
        super().__init__(datasource_type)

        self.json_path = json_path

        self.file_names: list[str] = self._get_file_list()

    def __iter__(self) -> Iterator[Molecule | Structure]:
        """Iterate over all entries in the datasource."""
        for i in range(len(self.file_names)):
            yield self[i]

    def __len__(self) -> int:
        """Return the total number of entries in the datasource."""
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Molecule | Structure:
        """Return the structure or molecule at the given index.

        Args:
            idx: Zero-based index into the datasource.

        Returns:
            The deserialised pymatgen ``Molecule`` or ``Structure`` at
            position ``idx``, with ``file_name`` stored in ``properties``.
        """
        file = self.file_names[idx]
        json_file = Path(self.json_path) / f"{file}.json"
        structure = self.load_json(json_file)
        structure.properties["file_name"] = file
        return structure

    def _get_file_list(self) -> list[str]:
        """Build the sorted list of JSON file stems in ``json_path``.

        Only files ending in ``.json`` are considered. Unrelated files are
        ignored.

        Returns:
            Sorted list of file stems found in ``json_path``.

        Raises:
            ResourceError: If ``json_path`` is not a directory or no JSON
                files are found in it.
        """
        json_path = Path(self.json_path)

        if not json_path.is_dir():
            raise ResourceError(f"JSON directory does not exist: {json_path}")

        file_names = sorted(list_filestems(json_path, suffixes=".json"))

        if not file_names:
            raise ResourceError(f"No JSON files found in directory: {json_path}")

        return file_names

    @staticmethod
    def load_json(json_file: Path) -> Molecule | Structure:
        """Load a pymatgen JSON file into a ``Structure`` or ``Molecule``.

        Dispatches on the ``@class`` key in the JSON object. Falls back to
        trying ``Structure`` then ``Molecule`` parsing when the key is absent
        or unsupported.

        Args:
            json_file: Path to the ``.json`` file.

        Returns:
            The deserialised pymatgen ``Molecule`` or ``Structure``.

        Raises:
            ResourceError: If the file cannot be parsed as a ``Structure`` or
                ``Molecule``, or if the JSON content is not an object.
        """
        with open(json_file, "r", encoding="utf-8") as f:
            entry = json.load(f)

        if not isinstance(entry, dict):
            raise ResourceError(
                f"Unsupported JSON content in {json_file}: " f"{type(entry).__name__}. Expected a JSON object."
            )

        class_name = entry.get("@class")

        if class_name == "Structure":
            return Structure.from_dict(entry)

        if class_name == "Molecule":
            return Molecule.from_dict(entry)

        if class_name is None:
            logging.warning(f"JSON file {json_file} is missing '@class' key. Attempting fallback parsing.")
        else:
            logging.warning(
                f"JSON file {json_file} has unsupported '@class' value {class_name!r}. Attempting fallback parsing."
            )

        # Fallback for valid pymatgen dicts missing @class.
        try:
            return Structure.from_dict(entry)
        except Exception:
            pass

        try:
            return Molecule.from_dict(entry)
        except Exception:
            pass

        raise ResourceError(
            f"Unsupported JSON object type in {json_file}: " f"{class_name!r}. Expected Structure or Molecule."
        )
