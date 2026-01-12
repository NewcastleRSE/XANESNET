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
    A class for reading the descriptor straight from a file. It tries to avoid
    doing any of the fancy stuff the other descriptors do. Only reads the file
    """

    def __init__(self, nfeatures):
        self.register_config(locals(), type="direct")
        self.nfeatures = nfeatures

    def transform(self, system: Atoms) -> np.ndarray:
        pass

    def get_nfeatures(self) -> int:
        return self.nfeatures

    def get_type(self) -> str:
        return "direct"
