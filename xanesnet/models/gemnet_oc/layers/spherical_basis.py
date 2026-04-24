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

import torch

from xanesnet.serialization.config import Config

from .basis_utils import get_sph_harm_basis
from .radial_basis import GaussianBasis, RadialBasis
from .scaling import ScaleFactor


class CircularBasisLayer(torch.nn.Module):
    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        cbf: Config,
        scale_basis: bool = False,
    ) -> None:
        super().__init__()
        self.radial_basis = radial_basis

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_cbf = ScaleFactor()

        cbf_name = cbf.get_str("name").lower()
        cbf_hparams = {k: v for k, v in cbf.as_dict().items() if k != "name"}
        if cbf_name == "gaussian":
            self.cosφ_basis = GaussianBasis(start=-1, stop=1, num_gaussians=num_spherical, **cbf_hparams)
        elif cbf_name == "spherical_harmonics":
            self.cosφ_basis = get_sph_harm_basis(num_spherical, zero_m_only=True)
        else:
            raise ValueError(f"Unknown cosine basis function '{cbf_name}'.")

    def forward(self, D_ca: torch.Tensor, cosφ_cab: torch.Tensor):
        rad_basis = self.radial_basis(D_ca)
        cir_basis = self.cosφ_basis(cosφ_cab)
        if self.scale_basis:
            cir_basis = self.scale_cbf(cir_basis)
        return rad_basis, cir_basis


class SphericalBasisLayer(torch.nn.Module):
    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        sbf: Config,
        scale_basis: bool = False,
    ) -> None:
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
            self.spherical_basis = lambda cosφ, ϑ: (
                circular_basis(cosφ)[:, :, None] * circular_basis(torch.cos(ϑ))[:, None, :]
            ).reshape(cosφ.shape[0], num_spherical**2)
        elif sbf_name == "gaussian_outer":
            self.circular_basis = GaussianBasis(start=-1, stop=1, num_gaussians=num_spherical, **sbf_hparams)
            self.spherical_basis = lambda cosφ, ϑ: (
                self.circular_basis(cosφ)[:, :, None] * self.circular_basis(torch.cos(ϑ))[:, None, :]
            ).reshape(cosφ.shape[0], num_spherical**2)
        else:
            raise ValueError(f"Unknown spherical basis function '{sbf_name}'.")

    def forward(self, D_ca: torch.Tensor, cosφ_cab: torch.Tensor, θ_cabd: torch.Tensor):
        rad_basis = self.radial_basis(D_ca)
        sph_basis = self.spherical_basis(cosφ_cab, θ_cabd)
        if self.scale_basis:
            sph_basis = self.scale_sbf(sph_basis)
        return rad_basis, sph_basis
