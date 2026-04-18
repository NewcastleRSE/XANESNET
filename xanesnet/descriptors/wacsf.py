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


@DescriptorRegistry.register("wacsf")
class WACSF(Descriptor):
    """
    A class for transforming a molecular system into a weighted atom-centered
    symmetry function (WACSF) descriptor. WACSFs encode the local geometry
    around a site using parameterised radial and angular components.

    References:
        J. Chem. Phys., 2018, 148, 241709 (10.1063/1.5019667)
        J. Chem. Phys., 2011, 134, 074106 (10.1063/1.3553717)
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
    ):
        """
        Args:
            r_min (float): Minimum radial cutoff distance (in A). Defaults to 1.0.
            r_max (float): Maximum radial cutoff distance (in A). Defaults to 6.0.
            n_g2 (int): Number of G2 symmetry functions. Defaults to 16.
            n_g4 (int): Number of G4 symmetry functions. Defaults to 32.
            l (list[float] | None): Lambda values for G4 encoding. Defaults to [1.0, -1.0].
            z (list[float] | None): Zeta values for G4 encoding. Defaults to [1.0].
            g2_parameterisation (str): G2 parameterisation strategy ('shifted' or 'centred').
            g4_parameterisation (str): G4 parameterisation strategy ('shifted' or 'centred').
            use_charge (bool): Append charge state to descriptor. Defaults to False.
            use_spin (bool): Append spin state to descriptor. Defaults to False.
        """
        super().__init__(descriptor_type)

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
        site_index: int | None = 0,
    ) -> np.ndarray:
        positions = system.get_positions()
        n_atoms = len(system)

        # Precompute pairwise distances (N x N)
        diff = positions[:, None, :] - positions[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        if site_index is not None:
            return self._transform_single(system, site_index, dist_matrix)

        return np.vstack([self._transform_single(system, i, dist_matrix) for i in range(n_atoms)])

    def _transform_single(
        self,
        system: Atoms,
        site_index: int,
        dist_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute the WACSF for a single site."""
        # Neighbour detection
        rij_all = dist_matrix[site_index]
        mask = (rij_all < self.r_max) & (rij_all > 0.0)
        neighbours = np.where(mask)[0]

        # If no neighbours, return zeros
        if len(neighbours) == 0:
            base_size = 1 + self.n_g2 + self.n_g4 + self.use_charge + self.use_spin
            return np.zeros(base_size)

        Z = 0.1 * system.get_atomic_numbers()

        # G1 term (radial cutoff sum)
        rij = rij_all[neighbours]
        g1 = np.sum(_cosine_cutoff(rij, self.r_max))
        features: list[np.ndarray] = [np.array([g1], dtype=float)]

        # G2 symmetry functions
        if self.n_g2:
            zj = Z[neighbours]
            cutoff_ij = _cosine_cutoff(rij, self.r_max)

            g2_vals = []
            for h, m in zip(self.g2_params.h, self.g2_params.m):
                gauss = np.exp(-h * (rij - m) ** 2)
                g2_vals.append(np.sum(zj * gauss * cutoff_ij))

            features.append(np.array(g2_vals))

        # G4 symmetry functions
        if self.n_g4:
            j_idx, k_idx = np.triu_indices(len(neighbours), k=1)
            j = neighbours[j_idx]
            k = neighbours[k_idx]

            rij = dist_matrix[site_index, j]
            rik = dist_matrix[site_index, k]
            rjk = dist_matrix[j, k]

            cutoff_ij = _cosine_cutoff(rij, self.r_max)
            cutoff_ik = _cosine_cutoff(rik, self.r_max)
            cutoff_jk = _cosine_cutoff(rjk, self.r_max)

            # Angles j-site-k
            pos = system.get_positions()
            vj = pos[j] - pos[site_index]
            vk = pos[k] - pos[site_index]

            dot = np.einsum("ij,ij->i", vj, vk)
            norms = np.linalg.norm(vj, axis=1) * np.linalg.norm(vk, axis=1)
            cosang = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0.0)
            cosang = np.clip(cosang, -1.0, 1.0)

            zj = Z[j]
            zk = Z[k]

            g4_vals = []
            for h, m in zip(self.g4_h, self.g4_m):
                gauss_ij = np.exp(-h * (rij - m) ** 2)
                gauss_ik = np.exp(-h * (rik - m) ** 2)
                gauss_jk = np.exp(-h * (rjk - m) ** 2)

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


###############################################################################
################################## HELPERS ####################################
###############################################################################


class _SymFuncParams:
    """
    Parameter container for symmetry function parameterisation.

    Computes eta (h) and mu (m) grids based on the 'shifted' or 'centred'
    strategy from Marquetand et al.; J. Chem. Phys., 2018, 148, 241709.
    """

    def __init__(self, n: int, r_min: float, r_max: float, parameterisation: str):
        self.n = n
        self.r_min = r_min
        self.r_max = r_max

        if parameterisation == "shifted":
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
    """Cosine cutoff function. See Behler; J. Chem. Phys., 2011, 134, 074106."""
    return (np.cos((np.pi * r) / r_max) + 1.0) / 2.0
