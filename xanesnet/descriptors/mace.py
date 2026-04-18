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
from mace.calculators import mace_mp

from .base import Descriptor
from .registry import DescriptorRegistry

###############################################################################
################################## CLASSES ####################################
###############################################################################


@DescriptorRegistry.register("mace")
class MACE(Descriptor):
    def __init__(
        self,
        descriptor_type: str,
        invariants_only: bool = False,
        num_layers: int = -1,
    ):
        super().__init__(descriptor_type)

        self.invariants_only = invariants_only
        self.num_layers = num_layers
        self.mace = mace_mp()

    def transform(
        self,
        system: Atoms,
        site_index: int | None = 0,
    ) -> np.ndarray:
        descriptors = np.asarray(
            self.mace.get_descriptors(
                system,
                invariants_only=self.invariants_only,
                num_layers=self.num_layers,
            )
        )
        if site_index is not None:
            return descriptors[site_index, :]
        return descriptors
