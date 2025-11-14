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

from typing import Dict

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np

from ase import Atoms
from abc import ABC
from abc import abstractmethod

from xanesnet.descriptors.vector_descriptor import VectorDescriptor
from xanesnet.registry import register_descriptor


###############################################################################
################################## CLASSES ####################################
###############################################################################


@register_descriptor("wacsf")
class WACSF(VectorDescriptor):
    """
    A class for transforming a molecular system into a weighted atom-centered
    symmetry function (WACSF) descriptor. WACSFs encode the local geometry
    around an absorption site using parameterised radial and angular
    components. For reference, check out the following publication:

    > J. Chem. Phys., 2018, 148, 241709 (10.1063/1.5019667)

    ...which builds on the earlier ACSF descriptor introduced in:

    > J. Chem. Phys., 2011, 134, 074106 (10.1063/1.3553717)
    """

    def __init__(
        self,
        r_min: float = 1.0,
        r_max: float = 6.0,
        n_g2: int = 16,
        n_g4: int = 32,
        l: list = None,
        z: list = None,
        g2_parameterisation: str = "shifted",
        g4_parameterisation: str = "centred",
        use_charge=False,
        use_spin=False,
    ):
        """
        Args:
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
            n_g2 (int, optional): The number of G2 symmetry functions to use
                for encoding.
                Defaults to 0.
            n_g4 (int, optional): The number of G4 symmetry functions to use
                for encoding.
                Defaults to 0.
            l (list, optional): List of lambda values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0, -1.0].
            z (list, optional): List of zeta values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0].
            g2_parameterisation (str, optional): The strategy to use for G2
                symmetry function parameterisation; choices are 'shifted' or
                'centred'. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to 'shifted'.
            g4_parameterisation (str, optional): The strategy to use for G4
                symmetry function parameterisation; choices are 'shifted' or
                'centred'. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to 'centred'.
            use_charge (bool): If True, includes an additional element in the
                vector descriptor for the charge state of the complex.
                Defaults to False.
            use_spin (bool): If True, includes an additional element in the
                vector descriptor for the spin state of the complex.
                Defaults to False.
        """

        super().__init__(r_min, r_max, use_charge, use_spin)

        self.register_config(locals(), type="wacsf")

        self.r_min = r_min
        self.r_max = r_max
        self.n_g2 = n_g2
        self.n_g4 = n_g4
        self.g2_parameterisation = g2_parameterisation
        self.g4_parameterisation = g4_parameterisation
        self.use_charge = use_charge
        self.use_spin = use_spin
        absorber_atom_only: bool = True,
        if self.n_g4:
            self.l = l if l is not None else [1.0, -1.0]
            self.z = z if z is not None else [1.0]

        if self.n_g2:
            self.g2_transformer = G2SymmetryFunctionTransformer(
                self.n_g2,
                r_min=self.r_min,
                r_max=self.r_max,
                parameterisation=self.g2_parameterisation,
            )

        if self.n_g4:
            self.g4_transformer = G4SymmetryFunctionTransformer(
                self.n_g4,
                r_min=self.r_min,
                r_max=self.r_max,
                l=self.l,
                z=self.z,
                parameterisation=self.g4_parameterisation,
            )

    def transform(self, system: Atoms) -> np.ndarray:
        positions = system.get_positions()
        n_atoms = len(system)
    
        # Precompute pairwise distances (N × N)
        diff = positions[:, None, :] - positions[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
    
        if self.absorber_atom_only:
            return self.transform_single_index(system, 0, dist_matrix)
    
        # Compute WACSF for all atoms
        wacsf_list = [
            self.transform_single_index(system, i, dist_matrix)
            for i in range(n_atoms)
        ]
    
        return np.vstack(wacsf_list)
        
    def transform_single_index(self, system: Atoms, index: int, dist_matrix: np.ndarray) -> np.ndarray:
    
        # -----------------------------
        # Neighbour detection
        # -----------------------------
        rij_all = dist_matrix[index]                     # distances from index → all atoms
        mask = (rij_all < self.r_max) & (rij_all > 0.0)  # exclude self
        neighbours = np.where(mask)[0]
    
        # If no neighbours, return zeros
        if len(neighbours) == 0:
            base_size = 1 + self.n_g2 + self.n_g4 + self.use_charge + self.use_spin
            return np.zeros(base_size)
    
        # Atomic numbers (scaled by 0.1 like your original code)
        Z = 0.1 * system.get_atomic_numbers()
    
        # -----------------------------
        # G1 term (radial cutoff sum)
        # -----------------------------
        rij = rij_all[neighbours]
        g1 = np.sum(cosine_cutoff(rij, self.r_max))
    
        features = [np.array([g1], dtype=float)]
    
        # -----------------------------
        # G2 symmetry functions
        # -----------------------------
        if self.n_g2:
            zj = Z[neighbours]
            cutoff_ij = cosine_cutoff(rij, self.r_max)
    
            g2_vals = []
            for h, m in zip(self.g2_transformer.h, self.g2_transformer.m):
                gauss = np.exp(-h * (rij - m) ** 2)
                g2_vals.append(np.sum(zj * gauss * cutoff_ij))
    
            features.append(np.array(g2_vals))
    
        # -----------------------------
        # G4 symmetry functions
        # -----------------------------
        if self.n_g4:
            j_idx, k_idx = np.triu_indices(len(neighbours), k=1)
            j = neighbours[j_idx]
            k = neighbours[k_idx]
    
            # Distances for pairs (all vectorized)
            rij = dist_matrix[index, j]
            rik = dist_matrix[index, k]
            rjk = dist_matrix[j, k]
    
            # Cutoffs
            cutoff_ij = cosine_cutoff(rij, self.r_max)
            cutoff_ik = cosine_cutoff(rik, self.r_max)
            cutoff_jk = cosine_cutoff(rjk, self.r_max)
    
            # Angles j–index–k
            # ajik shape = (#pairs,)
            pos = system.get_positions()
            vj = pos[j] - pos[index]
            vk = pos[k] - pos[index]
    
            dot = np.einsum('ij,ij->i', vj, vk)
            cosang = dot / (np.linalg.norm(vj, axis=1) * np.linalg.norm(vk, axis=1))
            ajik = np.arccos(np.clip(cosang, -1, 1))
    
            # Element weights
            zj = Z[j]
            zk = Z[k]
    
            # G4 full evaluation
            g4_vals = []
            i = 0
            for h, m in zip(self.g4_transformer.h, self.g4_transformer.m):
                gauss_ij = np.exp(-h * (rij - m) ** 2)
                gauss_ik = np.exp(-h * (rik - m) ** 2)
                gauss_jk = np.exp(-h * (rjk - m) ** 2)
    
                base_val = (
                    zj * zk *
                    gauss_ij * cutoff_ij *
                    gauss_ik * cutoff_ik *
                    gauss_jk * cutoff_jk
                )
    
                for lam in self.g4_transformer.l:
                    cos_term = 1.0 + lam * cosang
                    for zeta in self.g4_transformer.z:
                        g4_val = np.sum(base_val * (cos_term ** zeta)) * (2.0 ** (1 - zeta))
                        g4_vals.append(g4_val)
                        i += 1
    
            features.append(np.array(g4_vals))
        # -----------------------------
        # Optional spin / charge
        # -----------------------------
        if self.use_spin:
            features.append(np.array([system.info["S"]]))
    
        if self.use_charge:
            features.append(np.array([system.info["q"]]))
        return np.concatenate(features)
    
    def get_nfeatures(self) -> int:
        return int(1 + self.n_g2 + self.n_g4 + self.use_charge + self.use_spin)

    def get_type(self) -> str:
        return "wacsf"
        
#    def transform(self, system: Atoms) -> np.ndarray:
#        rij_in_range = system.get_distances(0, range(len(system))) < self.r_max
#        system = system[rij_in_range]
#
#        ij = np.array([[0, j] for j in range(1, len(system))], dtype="uint16")
#
#        if self.n_g4:
#            jik = np.array(
#                [
#                    [j, 0, k]
#                    for j in range(1, len(system))
#                    for k in range(1, len(system))
#                    if k > j
#                ],
#                dtype="uint16",
#            )
#
#        rij = system.get_distances(ij[:, 0], ij[:, 1])
#        g1 = np.sum(cosine_cutoff(rij, self.r_max))
#
#        wacsf = g1
#
#        if self.n_g2:
#            zj = system.get_atomic_numbers()[ij[:, 1]]
#            zj = 0.1 * zj
#            rij = system.get_distances(ij[:, 0], ij[:, 1])
#            g2 = self.g2_transformer.transform(zj, rij)
#            wacsf = np.append(wacsf, g2)
#
#        if self.n_g4:
#            zj = system.get_atomic_numbers()[jik[:, 0]]
#            zk = system.get_atomic_numbers()[jik[:, 2]]
#            zj = 0.1 * zj
#            zk = 0.1 * zk
#            rij = system.get_distances(jik[:, 1], jik[:, 0])
#            rik = system.get_distances(jik[:, 1], jik[:, 2])
#            rjk = system.get_distances(jik[:, 0], jik[:, 2])
#            ajik = np.radians(system.get_angles(jik))
#            g4 = self.g4_transformer.transform(zj, zk, rij, rik, rjk, ajik)
#            wacsf = np.append(wacsf, g4)
#
#        if self.use_spin:
#            wacsf = np.append(system.info["S"], wacsf)
#
#        if self.use_charge:
#            wacsf = np.append(system.info["q"], wacsf)
#
#        return wacsf
#
#    def get_nfeatures(self) -> int:
#        return int(1 + self.n_g2 + self.n_g4 + self.use_charge + self.use_spin)
#
#    def get_type(self) -> str:
#        return "wacsf"


class SymmetryFunctionTransformer(ABC):
    """
    An abstract base class for generating symmetry function vectors.
    """

    def __init__(self, n: int, r_min: float, r_max: float, parameterisation: str):
        """
        Args:
            n (int): The number of symmetry functions to use for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
            parameterisation (str, optional): The strategy to use for symmetry
                function parameterisation; choices are 'shifted' or 'centred'.
                For details, see Marquetand et al.; J. Chem. Phys., 2018, 148,
                241709 (DOI: 10.1063/1.5019667).
                Defaults to 'shifted'.
        """

        self.n = n
        self.r_min = r_min
        self.r_max = r_max
        self.parameterisation = parameterisation

        if self.parameterisation == "shifted":
            r_aux = np.linspace(self.r_min + 0.5, self.r_max - 0.5, self.n)
            dr = np.diff(r_aux)[0]
            self.h = np.array([1.0 / (2.0 * (dr**2)) for _ in r_aux])
            self.m = np.array([i for i in r_aux])
        elif self.parameterisation == "centred":
            r_aux = np.linspace(self.r_min + 1.0, self.r_max - 0.5, self.n)
            self.h = np.array([1.0 / (2.0 * (i**2)) for i in r_aux])
            self.m = np.array([0.0 for _ in r_aux])
        else:
            err_str = (
                "parameterisation options: 'shifted' | 'centred'; "
                "for details, see DOI: 10.1063/1.5019667"
            )
            raise ValueError(err_str)

    @abstractmethod
    def transform(self, *args) -> np.ndarray:
        """
        Encodes structural information (`*args`) using symmetry functions.

        Returns:
            np.ndarray: A symmetry function vector.
        """

        pass


class G2SymmetryFunctionTransformer(SymmetryFunctionTransformer):
    """
    A class for generating G2 symmetry function vectors.
    """

    def __init__(
        self,
        n: int,
        r_min: float = 0.0,
        r_max: float = 8.0,
        parameterisation: str = "shifted",
    ):
        """
        Args:
            n (int): The number of G2 symmetry functions to use for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
            parameterisation (str, optional): The strategy to use for symmetry
                function parameterisation; choices are 'shifted' or 'centred'.
                For details, see Marquetand et al.; J. Chem. Phys., 2018, 148,
                241709 (DOI: 10.1063/1.5019667).
                Defaults to 'shifted'.
        """

        super().__init__(n, r_min=r_min, r_max=r_max, parameterisation=parameterisation)

    def transform(self, zj: np.ndarray, rij: np.ndarray) -> np.ndarray:
        g2 = np.full(self.n, np.nan)

        cutoff_ij = cosine_cutoff(rij, self.r_max)

        i = 0
        for h, m in zip(self.h, self.m):
            g2[i] = np.sum(zj * gaussian(rij, h, m) * cutoff_ij)
            i += 1

        return g2


class G4SymmetryFunctionTransformer(SymmetryFunctionTransformer):
    """
    A class for generating G4 symmetry function vectors.
    """

    def __init__(
        self,
        n: int,
        r_min: float = 0.0,
        r_max: float = 8.0,
        l: list = [1.0, -1.0],
        z: list = [1.0],
        parameterisation: str = "centred",
    ):
        """
        Args:
            n (int): The number of G4 symmetry functions to use for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
            l (list, optional): List of lambda values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0, -1.0].
            z (list, optional): List of zeta values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0].
            parameterisation (str, optional): The strategy to use for symmetry
                function parameterisation; choices are 'shifted' or 'centred'.
                For details, see Marquetand et al.; J. Chem. Phys., 2018, 148,
                241709 (DOI: 10.1063/1.5019667).
                Defaults to 'centred'.
        """

        super().__init__(n, r_min=r_min, r_max=r_max, parameterisation=parameterisation)

        if self.n % (len(l) * len(z)):
            err_str = (
                f"can't generate {self.n} G4 symmetry functions with "
                f"{len(l)} lambda and {len(z)} zeta value(s)"
            )
            raise ValueError(err_str)
        else:
            n_ = int(self.n / (len(l) * len(z)))
            self.h = self.h[:n_]
            self.m = self.m[:n_]
            self.l = np.array(l)
            self.z = np.array(z)

    def transform(
        self,
        zj: np.ndarray,
        zk: np.ndarray,
        rij: np.ndarray,
        rik: np.ndarray,
        rjk: np.ndarray,
        ajik: np.ndarray,
    ) -> np.ndarray:
        g4 = np.full(self.n, np.nan)

        cutoff_ij = cosine_cutoff(rij, self.r_max)
        cutoff_ik = cosine_cutoff(rik, self.r_max)
        cutoff_jk = cosine_cutoff(rjk, self.r_max)

        i = 0
        for h, m in zip(self.h, self.m):
            for l in self.l:
                for z in self.z:
                    g4[i] = np.sum(
                        zj
                        * zk
                        * (1.0 + (l * np.cos(ajik))) ** z
                        * gaussian(rij, h, m)
                        * cutoff_ij
                        * gaussian(rik, h, m)
                        * cutoff_ik
                        * gaussian(rjk, h, m)
                        * cutoff_jk
                    ) * (2.0 ** (1.0 - z))
                    i += 1

        return g4


def cosine_cutoff(r: np.ndarray, r_max: float) -> np.ndarray:
    # returns a cosine cutoff function defined over `r` with `r_max` defining
    # the cutoff radius; see Behler; J. Chem. Phys., 2011, 134, 074106
    # (DOI: 10.1063/1.3553717)

    return (np.cos((np.pi * r) / r_max) + 1.0) / 2.0


def gaussian(r: np.ndarray, h: float, m: float) -> np.ndarray:
    # returns a gaussian-like function defined over `r` with eta (`h`) defining
    # the width and mu (`h`) defining the centrepoint/peak position

    return np.exp(-1.0 * h * (r - m) ** 2)
