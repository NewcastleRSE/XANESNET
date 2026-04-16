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

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from pymatgen.core import Molecule, Structure

from xanesnet.utils.exceptions import ResourceError
from xanesnet.utils.filesystem import list_filestems

from .base import DataSource
from .registry import DataSourceRegistry

###############################################################################
#################################### CLASS ####################################
###############################################################################


@DataSourceRegistry.register("pmgjson")
class PMGJSONSource(DataSource):
    """
    Datasource for pymatgen JSON files in a single directory.
    Each JSON file is expected to contain a single structure or molecule entry.
    """

    def __init__(
        self,
        datasource_type: str,
        json_path: str,
    ) -> None:
        super().__init__(datasource_type)

        self.json_path = json_path

        self.file_names: list[str] = self._get_file_list()

    def __iter__(self) -> Iterator[Molecule | Structure]:
        for file in self.file_names:
            json_file = Path(self.json_path) / f"{file}.json"
            structure = self.load_json(json_file)
            structure.properties["file_name"] = file
            yield structure

    def __len__(self) -> int:
        return len(self.file_names)

    def _get_file_list(self) -> list[str]:
        """
        Get list of file stems in the directory.
        """
        json_path = Path(self.json_path)

        json_stems = list(list_filestems(json_path))
        file_names = sorted(json_stems)

        if not file_names:
            raise ResourceError(f"No JSON files found in directory: {json_path}")

        return file_names

    @staticmethod
    def load_json(json_file: Path) -> Molecule | Structure:
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

        logging.warning(f"JSON file {json_file} is missing '@class' key. Attempting fallback parsing.")

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
