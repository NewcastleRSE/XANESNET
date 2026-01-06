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

from xanesnet.descriptors.base_descriptor import BaseDescriptor
from xanesnet.registry import DescriptorRegistry

###############################################################################
################################## CLASSES ####################################
###############################################################################


@DescriptorRegistry.register("mace")
class MACE(BaseDescriptor):
    def __init__(
        self,
        invariants_only: bool = False,
        num_layers: int = -1,
        absorber_atom_only: bool = False,
    ):
        self.register_config(locals(), type="mace")
        self.invariants_only = invariants_only
        self.num_layers = num_layers
        self.absorber_atom_only = absorber_atom_only
        self.mace = mace_mp()

    def transform(self, system: Atoms) -> np.ndarray:
        tmp = self.mace.get_descriptors(system, invariants_only=self.invariants_only, num_layers=self.num_layers)
        if self.absorber_atom_only:
            return tmp[0, :]
        else:
            return tmp

    def get_nfeatures(self) -> int:
        num_interactions = int(self.mace.models[0].num_interactions)
        if self.num_layers == -1:
            self.num_layers = num_interactions
        elif self.num_layers > num_interactions + 1:
            raise ValueError("num_layers cannot be greater than num_interactions+1")

        irreps_out = self.mace.models[0].products[0].linear.__dict__["irreps_out"]
        l_max = irreps_out.lmax
        num_features = irreps_out.dim // (l_max + 1) ** 2

        if self.invariants_only:
            total = 0
            for i in range(self.num_layers - 1):
                total += (i * (l_max + 1) ** 2 + 1) * num_features - i * (l_max + 1) ** 2 * num_features
            n_feats = num_features + total
        else:
            n_feats = ((num_interactions - 1) * (l_max + 1) ** 2 + 1) * num_features
        return n_feats

    def get_type(self) -> str:
        return "mace"
