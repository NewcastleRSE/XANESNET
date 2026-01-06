"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

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

from xanesnet.descriptors.vector_descriptor import VectorDescriptor
from xanesnet.registry import DescriptorRegistry

###############################################################################
################################## CLASSES ####################################
###############################################################################


@DescriptorRegistry.register("rdc")
class RDC(VectorDescriptor):
    """
    A class for transforming a molecular system into a radial (or 'pair')
    distribution curve (RDCs). The RDC is - simplistically - like a histogram
    of pairwise internuclear distances discretised over an auxilliary
    real-space grid and smoothed out using Gaussians; pairs are made between
    the absorption site and all atoms within a defined radial cutoff.
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 8.0,
        dr: float = 0.01,
        alpha: float = 10.0,
        use_charge=False,
        use_spin=False,
    ):
        """
        Args:
            r_min (float): The minimum radial cutoff distance (in A) around
                the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float): The maximum radial cutoff distance (in A) around
                the absorption site.
                Defaults to 6.0.
            dr (float): The step size (in A) for the auxilliary real-space grid
                that the RDC is discretised over.
                Defaults to 0.01.
            alpha (float): A smoothing parameter used in a Gaussian exponent
                that defines the effective spatial resolution of the RDC.
                Defaults to 10.0.
            use_charge (bool): If True, includes an additional element in the
                vector descriptor for the charge state of the complex.
                Defaults to False.
            use_spin (bool): If True, includes an additional element in the
                vector descriptor for the spin state of the complex.
                Defaults to False.
        """

        super().__init__(r_min, r_max, use_charge, use_spin)

        self.register_config(locals(), type="rdc")

        if isinstance(dr, (int, float)) and r_max >= dr > 0.0:
            self.dr = float(dr)
        else:
            raise ValueError(f"expected dr: int/float > 0.0; got {dr}")

        if isinstance(alpha, (int, float)) and alpha > 0.0:
            self.alpha = float(alpha)
        else:
            raise ValueError(f"expected alpha: int/float > 0.0; got {alpha}")

        nr_aux = int(np.absolute(self.r_max - self.r_min) / self.dr) + 1
        self.r_aux = np.linspace(self.r_min, self.r_max, nr_aux)

    def transform(self, system: Atoms) -> np.ndarray:
        if not isinstance(system, Atoms):
            raise TypeError(
                f"systems passed as arguments to .transform ",
                "should be ase.Atoms objects",
            )

        rij_in_range = system.get_distances(0, range(len(system))) < self.r_max
        system = system[rij_in_range]

        ij = [[0, j] for j in range(1, len(system))]
        if len(ij) < 1:
            raise RuntimeError(
                f"too few atoms within {self.r_max:.2f} A of ",
                "the absorption site to set up non-zero radial distribution ",
                "curve (no pairs)",
            )
        else:
            ij = np.array(ij, dtype="uint16")

        zi = system.get_atomic_numbers()[ij[:, 0]]
        zj = system.get_atomic_numbers()[ij[:, 1]]
        rij = system.get_distances(ij[:, 0], ij[:, 1])
        rij_r_sq = np.square(rij[:, np.newaxis] - self.r_aux)
        exp = np.exp(-1.0 * self.alpha * rij_r_sq)
        rdc = np.sum((zi * zj)[:, np.newaxis] * exp, axis=0)

        if self.use_spin:
            rdc = np.append(system.info["S"], rdc)

        if self.use_charge:
            rdc = np.append(system.info["q"], rdc)

        return rdc

    def get_nfeatures(self) -> int:
        return len(self.r_aux) + self.use_charge + self.use_spin

    def get_type(self) -> str:
        return "rdc"
