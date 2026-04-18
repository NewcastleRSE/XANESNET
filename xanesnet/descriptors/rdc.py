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
from ase.neighborlist import neighbor_list

from .base import Descriptor
from .registry import DescriptorRegistry

###############################################################################
################################## CLASSES ####################################
###############################################################################


@DescriptorRegistry.register("rdc")
class RDC(Descriptor):
    """
    A class for transforming a molecular system into a radial (or 'pair')
    distribution curve (RDC). The RDC is a histogram of pairwise internuclear
    distances discretised over an auxiliary real-space grid and smoothed using
    Gaussians; pairs are made between a given site and all atoms within a
    defined radial cutoff.
    """

    def __init__(
        self,
        descriptor_type: str,
        r_min: float = 0.0,
        r_max: float = 8.0,
        dr: float = 0.01,
        alpha: float = 10.0,
        use_charge: bool = False,
        use_spin: bool = False,
    ):
        """
        Args:
            r_min (float): The minimum radial cutoff distance (in A).
                Defaults to 0.0.
            r_max (float): The maximum radial cutoff distance (in A).
                Defaults to 8.0.
            dr (float): The step size (in A) for the auxiliary real-space grid.
                Defaults to 0.01.
            alpha (float): A smoothing parameter for the Gaussian exponent.
                Defaults to 10.0.
            use_charge (bool): If True, appends the charge state to the descriptor.
                Defaults to False.
            use_spin (bool): If True, appends the spin state to the descriptor.
                Defaults to False.
        """
        super().__init__(descriptor_type)

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.dr = float(dr)
        self.alpha = float(alpha)
        self.use_charge = use_charge
        self.use_spin = use_spin

        nr_aux = int(np.absolute(self.r_max - self.r_min) / self.dr) + 1
        self.r_aux = np.linspace(self.r_min, self.r_max, nr_aux)

    def transform(self, system: Atoms, site_index: int | list[int] | None = 0) -> np.ndarray:
        if isinstance(site_index, int):
            site_index = [site_index]

        # Use ASE neighbor_list for correct periodic image enumeration.
        # mic=True only finds the closest image per atom, which is wrong
        # for small unit cells where r_max exceeds the cell dimensions.
        i_arr, j_arr, d_arr = neighbor_list("ijd", system, cutoff=self.r_max)

        indices = range(len(system)) if site_index is None else site_index
        return np.vstack([self._transform_single(system, idx, i_arr, j_arr, d_arr) for idx in indices])

    def _transform_single(
        self,
        system: Atoms,
        site_index: int,
        i_arr: np.ndarray,
        j_arr: np.ndarray,
        d_arr: np.ndarray,
    ) -> np.ndarray:
        """Compute the RDC for a single site."""
        mask = i_arr == site_index

        if mask.sum() < 1:
            raise RuntimeError(
                f"Too few atoms within {self.r_max:.2f} A of site {site_index} "
                "to compute a non-zero radial distribution curve."
            )

        zi = system.get_atomic_numbers()[site_index]
        zj = system.get_atomic_numbers()[j_arr[mask]]
        rij = d_arr[mask]

        rij_r_sq = np.square(rij[:, np.newaxis] - self.r_aux)
        exp = np.exp(-1.0 * self.alpha * rij_r_sq)
        rdc = np.sum((zi * zj)[:, np.newaxis] * exp, axis=0)

        if self.use_spin:
            rdc = np.append(system.info["S"], rdc)

        if self.use_charge:
            rdc = np.append(system.info["q"], rdc)

        return rdc
