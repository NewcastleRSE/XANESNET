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

import numpy as np
from ase import Atoms

from .base import Descriptor
from .registry import DescriptorRegistry

###############################################################################
################################## CLASSES ####################################
###############################################################################


@DescriptorRegistry.register("direct")
class DIRECT(Descriptor):
    """
    A descriptor that reads features straight from a file without any transformation.
    """

    # TODO NOT IMPLEMENTED YET

    def __init__(
        self,
        descriptor_type: str,
    ) -> None:
        super().__init__(descriptor_type)

        raise NotImplementedError("DIRECT descriptor not implemented yet.")

    def transform(
        self,
        system: Atoms,
        site_index: int | None = 0,
    ) -> np.ndarray:
        raise NotImplementedError("DIRECT descriptor not implemented yet.")
