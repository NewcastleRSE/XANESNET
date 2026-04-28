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

"""Abstract base class for all XANESNET data sources."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from pymatgen.core import Molecule, Structure


class DataSource(ABC):
    """Abstract base class for all XANESNET data sources.

    Args:
        datasource_type: Identifier string for the concrete datasource type.
    """

    def __init__(
        self,
        datasource_type: str,
    ) -> None:
        self.datasource_type = datasource_type

    @abstractmethod
    def __iter__(self) -> Iterator[Molecule | Structure]:
        """Iterate over all entries in the datasource.

        Returns:
            Iterator over the datasource entries.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of entries in the datasource.

        Returns:
            Number of available datasource entries.
        """
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Molecule | Structure:
        """Return the entry at the given index.

        Args:
            idx: Zero-based index into the datasource.

        Returns:
            The pymatgen ``Molecule`` or ``Structure`` at position ``idx``.
        """
        ...
