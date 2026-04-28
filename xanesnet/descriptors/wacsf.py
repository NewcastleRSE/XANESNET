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

"""Weighted atom-centered symmetry function (WACSF) descriptor for XANESNET."""

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from .base import Descriptor
from .registry import DescriptorRegistry


@DescriptorRegistry.register("wacsf")
class WACSF(Descriptor):
    """Weighted atom-centered symmetry function (WACSF) descriptor.

    Encodes the local geometry around a site using parameterised radial (G2)
    and angular (G4) symmetry functions with atom-type weighting.

    References:
        J. Chem. Phys., 2018, 148, 241709 (DOI: 10.1063/1.5019667)
        J. Chem. Phys., 2011, 134, 074106 (DOI: 10.1063/1.3553717)

    Args:
        descriptor_type: Identifier string for this descriptor type.
        r_min: Minimum radial distance. **A**. Defaults to ``1.0``.
        r_max: Maximum radial cutoff distance. **A**. Defaults to ``6.0``.
        n_g2: Number of G2 (radial) symmetry functions. Defaults to ``16``.
        n_g4: Number of G4 (angular) symmetry functions. Defaults to ``32``.
        l: Lambda values for G4 encoding. Defaults to ``[1.0, -1.0]``.
        z: Zeta values for G4 encoding. Defaults to ``[1.0]``.
        g2_parameterisation: G2 grid strategy — ``'shifted'`` or ``'centred'``.
            Defaults to ``'shifted'``.
        g4_parameterisation: G4 grid strategy — ``'shifted'`` or ``'centred'``.
            Defaults to ``'centred'``.
        use_charge: Append charge state scalar to the descriptor. Defaults to ``False``.
        use_spin: Append spin state scalar to the descriptor. Defaults to ``False``.
    """

    def __init__(
        self,
        descriptor_type: str,
        r_min: float = 1.0,
        r_max: float = 6.0,
        n_g2: int = 16,
        n_g4: int = 32,
        l: list[float] | None = None,
        z: list[float] | None = None,
        g2_parameterisation: str = "shifted",
        g4_parameterisation: str = "centred",
        use_charge: bool = False,
        use_spin: bool = False,
    ) -> None:
        super().__init__(descriptor_type)

        if n_g2 < 0:
            raise ValueError(f"n_g2 must be non-negative, got {n_g2}")
        if n_g4 < 0:
            raise ValueError(f"n_g4 must be non-negative, got {n_g4}")

        self.r_min = r_min
        self.r_max = r_max
        self.n_g2 = n_g2
        self.n_g4 = n_g4
        self.g2_parameterisation = g2_parameterisation
        self.g4_parameterisation = g4_parameterisation
        self.use_charge = use_charge
        self.use_spin = use_spin

        if self.n_g2:
            self.g2_params = _SymFuncParams(
                self.n_g2,
                r_min=self.r_min,
                r_max=self.r_max,
                parameterisation=self.g2_parameterisation,
            )

        if self.n_g4:
            l_vals = l if l is not None else [1.0, -1.0]
            z_vals = z if z is not None else [1.0]

            base_params = _SymFuncParams(
                self.n_g4,
                r_min=self.r_min,
                r_max=self.r_max,
                parameterisation=self.g4_parameterisation,
            )

            if self.n_g4 % (len(l_vals) * len(z_vals)):
                raise ValueError(
                    f"Can't generate {self.n_g4} G4 symmetry functions with "
                    f"{len(l_vals)} lambda and {len(z_vals)} zeta value(s)"
                )

            n_ = self.n_g4 // (len(l_vals) * len(z_vals))
            self.g4_h = base_params.h[:n_]
            self.g4_m = base_params.m[:n_]
            self.g4_l = np.array(l_vals)
            self.g4_z = np.array(z_vals)

    def transform(
        self,
        system: Atoms,
        site_index: int | list[int] | None = 0,
    ) -> np.ndarray:
        """Compute WACSF descriptors for one or more sites.

        Args:
            system: The atomic system.
            site_index: Site index, list of site indices, or ``None`` for all sites.
                Defaults to ``0`` (the absorber site).

        Returns:
            Descriptor array ``(S, F)`` where ``S`` is the number of selected sites
            and ``F = 1 + n_g2 + n_g4 + use_charge + use_spin``.
        """
        # Use ASE neighbor_list for correct periodic image enumeration.
        # get_all_distances(mic=True) only finds the closest image per atom,
        # which is wrong for small unit cells where r_max exceeds cell dimensions.
        i_arr, j_arr, d_arr, D_arr = neighbor_list("ijdD", system, cutoff=self.r_max)

        if isinstance(site_index, int):
            site_index = [site_index]

        indices = range(len(system)) if site_index is None else site_index
        return np.vstack([self._transform_single(system, idx, i_arr, j_arr, d_arr, D_arr) for idx in indices])

    def _transform_single(
        self,
        system: Atoms,
        site_index: int,
        i_arr: np.ndarray,
        j_arr: np.ndarray,
        d_arr: np.ndarray,
        D_arr: np.ndarray,
    ) -> np.ndarray:
        """Compute the WACSF fingerprint for a single absorber site.

        Args:
            system: The atomic system.
            site_index: Index of the absorber site.
            i_arr: Source atom indices from the neighbor list ``(P,)``.
            j_arr: Target atom indices from the neighbor list ``(P,)``.
            d_arr: Pairwise distances from the neighbor list ``(P,)``. **A**.
            D_arr: Displacement vectors from the neighbor list ``(P, 3)``. **A**.

        Returns:
            WACSF feature vector ``(F,)`` where ``F = 1 + n_g2 + n_g4 + use_charge + use_spin``.
            If no neighbours are found within ``r_max``, the geometric terms are
            zero while optional spin and charge scalars are still preserved.
        """
        mask = i_arr == site_index

        # If no neighbours, return zeroed geometric terms but preserve optional global scalars.
        if mask.sum() == 0:
            features: list[np.ndarray] = [np.zeros(1 + self.n_g2 + self.n_g4)]
            if self.use_spin:
                features.append(np.array([system.info["S"]]))
            if self.use_charge:
                features.append(np.array([system.info["q"]]))
            return np.concatenate(features)

        Z = 0.1 * system.get_atomic_numbers()

        j_neigh = j_arr[mask]
        rij = d_arr[mask]
        Dij = D_arr[mask]  # displacement vectors to each periodic image

        # G1 term (radial cutoff sum)
        g1 = np.sum(_cosine_cutoff(rij, self.r_max))
        features: list[np.ndarray] = [np.array([g1], dtype=float)]

        # G2 symmetry functions
        if self.n_g2:
            zj = Z[j_neigh]
            cutoff_ij = _cosine_cutoff(rij, self.r_max)

            g2_vals = []
            for h, m in zip(self.g2_params.h, self.g2_params.m):
                gauss = np.exp(-h * (rij - m) ** 2)
                g2_vals.append(np.sum(zj * gauss * cutoff_ij))

            features.append(np.array(g2_vals))

        # G4 symmetry functions
        if self.n_g4:
            n_neigh = len(j_neigh)
            jj, kk = np.triu_indices(n_neigh, k=1)

            r_ij = rij[jj]
            r_ik = rij[kk]
            # j-k distance from displacement vectors (correct for specific periodic images)
            r_jk = np.linalg.norm(Dij[kk] - Dij[jj], axis=1)

            cutoff_ij = _cosine_cutoff(r_ij, self.r_max)
            cutoff_ik = _cosine_cutoff(r_ik, self.r_max)
            cutoff_jk = _cosine_cutoff(r_jk, self.r_max)

            # Angles j-site-k from displacement vectors
            v_ij = Dij[jj]
            v_ik = Dij[kk]

            dot = np.einsum("ij,ij->i", v_ij, v_ik)
            norms = np.linalg.norm(v_ij, axis=1) * np.linalg.norm(v_ik, axis=1)
            cosang = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0.0)
            cosang = np.clip(cosang, -1.0, 1.0)

            zj = Z[j_neigh[jj]]
            zk = Z[j_neigh[kk]]

            g4_vals = []
            for h, m in zip(self.g4_h, self.g4_m):
                gauss_ij = np.exp(-h * (r_ij - m) ** 2)
                gauss_ik = np.exp(-h * (r_ik - m) ** 2)
                gauss_jk = np.exp(-h * (r_jk - m) ** 2)

                base_val = zj * zk * gauss_ij * cutoff_ij * gauss_ik * cutoff_ik * gauss_jk * cutoff_jk

                for lam in self.g4_l:
                    cos_term = 1.0 + lam * cosang
                    for zeta in self.g4_z:
                        g4_val = np.sum(base_val * (cos_term**zeta)) * (2.0 ** (1.0 - zeta))
                        g4_vals.append(g4_val)

            features.append(np.array(g4_vals))

        # Optional spin / charge
        if self.use_spin:
            features.append(np.array([system.info["S"]]))

        if self.use_charge:
            features.append(np.array([system.info["q"]]))

        return np.concatenate(features)


class _SymFuncParams:
    """Precomputed eta (``h``) and mu (``m``) grids for G2 / G4 symmetry functions.

    Computes parameter grids based on the ``'shifted'`` or ``'centred'``
    parameterisation from Marquetand et al.;
    J. Chem. Phys., 2018, 148, 241709 (DOI: 10.1063/1.5019667).

    Args:
        n: Number of symmetry functions.
        r_min: Minimum radial distance. **A**.
        r_max: Maximum radial distance. **A**.
        parameterisation: Grid strategy — ``'shifted'`` or ``'centred'``.

    Raises:
        ValueError: If ``n < 1`` or ``parameterisation`` is unsupported.
    """

    def __init__(self, n: int, r_min: float, r_max: float, parameterisation: str) -> None:
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")

        self.n = n
        self.r_min = r_min
        self.r_max = r_max

        if parameterisation == "shifted":
            if n == 1:
                dr = max(r_max - r_min, np.finfo(float).eps)
                self.h = np.array([1.0 / (2.0 * dr**2)])
                self.m = np.array([(r_min + r_max) / 2.0])
            else:
                r_aux = np.linspace(r_min + 0.5, r_max - 0.5, n)
                dr = np.diff(r_aux)[0]
                self.h = np.full(n, 1.0 / (2.0 * dr**2))
                self.m = r_aux.copy()
        elif parameterisation == "centred":
            r_aux = np.linspace(r_min + 1.0, r_max - 0.5, n)
            self.h = np.array([1.0 / (2.0 * r**2) for r in r_aux])
            self.m = np.zeros(n)
        else:
            raise ValueError(
                f"parameterisation must be 'shifted' or 'centred', got '{parameterisation}'. "
                "See DOI: 10.1063/1.5019667"
            )


def _cosine_cutoff(r: np.ndarray, r_max: float) -> np.ndarray:
    """Cosine cutoff function.

    Returns the standard Behler cosine cutoff inside ``[0, r_max]`` and zero
    outside the cutoff radius.

    Args:
        r: Distances at which to evaluate the cutoff.
        r_max: Cutoff radius.

    Returns:
        Cutoff values with the same shape as ``r``.
    """
    values = np.zeros_like(r, dtype=float)
    mask = r <= r_max
    values[mask] = (np.cos((np.pi * r[mask]) / r_max) + 1.0) / 2.0
    return values
