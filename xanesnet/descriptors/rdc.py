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

"""Radial distribution curve (RDC) descriptor for XANESNET."""

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from .base import Descriptor
from .registry import DescriptorRegistry


@DescriptorRegistry.register("rdc")
class RDC(Descriptor):
    """Radial distribution curve (RDC) descriptor.

    Transforms an atomic system into a histogram of pairwise internuclear
    distances, discretised over an auxiliary real-space grid and smoothed
    with Gaussians. Pairs are formed between a target site and all atoms
    within the radial cutoff.

    Args:
        descriptor_type: Identifier string for this descriptor type.
        r_min: Minimum radial grid distance. **A**. Defaults to ``0.0``.
        r_max: Maximum radial cutoff distance. **A**. Defaults to ``8.0``.
        dr: Grid spacing of the auxiliary real-space grid. **A**. Defaults to ``0.01``.
        alpha: Gaussian exponent (smoothing parameter). Defaults to ``10.0``.
        use_charge: Append the charge state scalar to the descriptor. Defaults to ``False``.
        use_spin: Append the spin state scalar to the descriptor. Defaults to ``False``.
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
    ) -> None:
        """Initialise the RDC descriptor.

        Args:
            descriptor_type: Identifier string for this descriptor type.
            r_min: Minimum radial grid distance. **A**. Defaults to ``0.0``.
            r_max: Maximum radial cutoff distance. **A**. Defaults to ``8.0``.
            dr: Grid spacing of the auxiliary real-space grid. **A**. Defaults to ``0.01``.
            alpha: Gaussian exponent (smoothing parameter). Defaults to ``10.0``.
            use_charge: Append the charge state scalar to the descriptor. Defaults to ``False``.
            use_spin: Append the spin state scalar to the descriptor. Defaults to ``False``.

        Raises:
            ValueError: If ``dr <= 0`` or ``r_max < r_min``.
        """
        super().__init__(descriptor_type)

        if dr <= 0:
            raise ValueError(f"dr must be positive, got {dr}")
        if r_max < r_min:
            raise ValueError(f"r_max must be greater than or equal to r_min, got r_min={r_min}, r_max={r_max}")

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.dr = float(dr)
        self.alpha = float(alpha)
        self.use_charge = use_charge
        self.use_spin = use_spin

        nr_aux = int(np.absolute(self.r_max - self.r_min) / self.dr) + 1
        self.r_aux = np.linspace(self.r_min, self.r_max, nr_aux)

    def transform(self, system: Atoms, site_index: int | list[int] | None = 0) -> np.ndarray:
        """Compute RDC descriptors for one or more sites.

        Args:
            system: The atomic system.
            site_index: Site index, list of site indices, or ``None`` for all sites.
                Defaults to ``0`` (the absorber site).

        Returns:
            Descriptor array ``(S, G)`` where ``S`` is the number of selected sites
            and ``G`` is the number of grid points (plus optional spin/charge scalars
            appended in that order).
        """
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
        """Compute the RDC fingerprint for a single absorber site.

        Args:
            system: The atomic system.
            site_index: Index of the absorber site.
            i_arr: Source atom indices from the neighbor list ``(P,)``.
            j_arr: Target atom indices from the neighbor list ``(P,)``.
            d_arr: Pairwise distances from the neighbor list ``(P,)``. **A**.

        Returns:
            RDC feature vector ``(G,)`` extended by optional spin/charge scalars
            appended in that order.

        Raises:
            RuntimeError: If no atoms lie within ``r_max`` of ``site_index``.
        """
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
            rdc = np.append(rdc, system.info["S"])

        if self.use_charge:
            rdc = np.append(rdc, system.info["q"])

        return rdc
