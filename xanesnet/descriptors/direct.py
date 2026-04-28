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

"""Pass-through descriptor that uses pre-computed features directly."""

import numpy as np
from ase import Atoms

from .base import Descriptor
from .registry import DescriptorRegistry


@DescriptorRegistry.register("direct")
class DIRECT(Descriptor):
    """Descriptor that reads pre-computed features directly without transformation.

    Args:
        descriptor_type: Identifier string for this descriptor type.
    """

    # TODO NOT IMPLEMENTED YET

    def __init__(
        self,
        descriptor_type: str,
    ) -> None:
        """Initialize ``DIRECT``."""
        super().__init__(descriptor_type)

        raise NotImplementedError("DIRECT descriptor not implemented yet.")

    def transform(
        self,
        system: Atoms,
        site_index: int | list[int] | None = 0,
    ) -> np.ndarray:
        """Raise ``NotImplementedError`` because the descriptor is not implemented.

        Args:
            system: The atomic system.
            site_index: Site index, list of site indices, or ``None`` for all sites.

        Returns:
            Precomputed descriptor array once implemented.

        Raises:
            NotImplementedError: Always, because ``DIRECT`` is not implemented yet.
        """
        raise NotImplementedError("DIRECT descriptor not implemented yet.")
