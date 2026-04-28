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

"""Circular and spherical basis layers for GemNet-OC angular embeddings."""

import torch

from xanesnet.serialization.config import Config

from .basis_utils import get_sph_harm_basis
from .radial_basis import GaussianBasis, RadialBasis
from .scaling import ScaleFactor


class CircularBasisLayer(torch.nn.Module):
    """Circular basis combining a radial basis and an angular (cosine) basis.

    Used for triplet interactions that only require the angle between two
    neighboring edges.

    Args:
        num_spherical: Number of angular basis functions.
        radial_basis: Pre-built :class:`RadialBasis` instance.
        cbf: Config specifying the circular basis type
            (``"gaussian"`` or ``"spherical_harmonics"``) and its
            hyperparameters.
        scale_basis: Apply a learnable :class:`ScaleFactor` to the angular
            component after evaluation.
    """

    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        cbf: Config,
        scale_basis: bool = False,
    ) -> None:
        """Initialize ``CircularBasisLayer``."""
        super().__init__()
        self.radial_basis = radial_basis

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_cbf = ScaleFactor()

        cbf_name = cbf.get_str("name").lower()
        cbf_hparams = {k: v for k, v in cbf.as_dict().items() if k != "name"}
        if cbf_name == "gaussian":
            self.cos_phi_basis = GaussianBasis(start=-1, stop=1, num_gaussians=num_spherical, **cbf_hparams)
        elif cbf_name == "spherical_harmonics":
            self.cos_phi_basis = get_sph_harm_basis(num_spherical, zero_m_only=True)
        else:
            raise ValueError(f"Unknown cosine basis function '{cbf_name}'.")

    def forward(self, D_ca: torch.Tensor, cos_phi_cab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the radial and circular basis for triplet interactions.

        Args:
            D_ca: Edge distances, shape ``(nEdges,)``.
            cos_phi_cab: Cosine of the triplet angle, shape ``(nTriplets,)``.

        Returns:
            Tuple of ``(rad_basis, cir_basis)`` with shapes
            ``(nEdges, num_radial)`` and ``(nTriplets, num_spherical)``.
        """
        rad_basis = self.radial_basis(D_ca)
        cir_basis = self.cos_phi_basis(cos_phi_cab)
        if self.scale_basis:
            cir_basis = self.scale_cbf(cir_basis)
        return rad_basis, cir_basis


class SphericalBasisLayer(torch.nn.Module):
    """Spherical basis combining a radial basis and a full angular (spherical harmonic) basis.

    Used for quadruplet interactions that require both the planar angle and the
    dihedral angle.

    Args:
        num_spherical: Number of spherical basis functions per angular
            dimension; output size is ``num_spherical ** 2`` for outer-product
            variants.
        radial_basis: Pre-built :class:`RadialBasis` instance.
        sbf: Config specifying the spherical basis type
            (``"spherical_harmonics"``, ``"legendre_outer"``, or
            ``"gaussian_outer"``) and its hyperparameters.
        scale_basis: Apply a learnable :class:`ScaleFactor` to the spherical
            component after evaluation.
    """

    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        sbf: Config,
        scale_basis: bool = False,
    ) -> None:
        """Initialize ``SphericalBasisLayer``."""
        super().__init__()
        self.num_spherical = num_spherical
        self.radial_basis = radial_basis

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_sbf = ScaleFactor()

        sbf_name = sbf.get_str("name").lower()
        sbf_hparams = {k: v for k, v in sbf.as_dict().items() if k != "name"}

        if sbf_name == "spherical_harmonics":
            self.spherical_basis = get_sph_harm_basis(num_spherical, zero_m_only=False)
        elif sbf_name == "legendre_outer":
            circular_basis = get_sph_harm_basis(num_spherical, zero_m_only=True)
            self.spherical_basis = lambda cos_phi, theta: (
                circular_basis(cos_phi)[:, :, None] * circular_basis(torch.cos(theta))[:, None, :]
            ).reshape(cos_phi.shape[0], num_spherical**2)
        elif sbf_name == "gaussian_outer":
            self.circular_basis = GaussianBasis(start=-1, stop=1, num_gaussians=num_spherical, **sbf_hparams)
            self.spherical_basis = lambda cos_phi, theta: (
                self.circular_basis(cos_phi)[:, :, None] * self.circular_basis(torch.cos(theta))[:, None, :]
            ).reshape(cos_phi.shape[0], num_spherical**2)
        else:
            raise ValueError(f"Unknown spherical basis function '{sbf_name}'.")

    def forward(
        self, D_ca: torch.Tensor, cos_phi_cab: torch.Tensor, theta_cabd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the radial and spherical basis for quadruplet interactions.

        Args:
            D_ca: Edge distances, shape ``(nEdges,)``.
            cos_phi_cab: Cosine of the triplet angle, shape ``(nQuadruplets,)``.
            theta_cabd: Dihedral angle in **radians**, shape ``(nQuadruplets,)``.

        Returns:
            Tuple of ``(rad_basis, sph_basis)`` with shapes
            ``(nEdges, num_radial)`` and ``(nQuadruplets, num_spherical**2)``
            (exact second dimension depends on the basis type).
        """
        rad_basis = self.radial_basis(D_ca)
        sph_basis = self.spherical_basis(cos_phi_cab, theta_cabd)
        if self.scale_basis:
            sph_basis = self.scale_sbf(sph_basis)
        return rad_basis, sph_basis
