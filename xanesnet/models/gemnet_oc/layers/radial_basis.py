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

"""Radial basis functions and envelope modules for GemNet-OC."""

import math

import numpy as np
import torch
from scipy.special import binom

from xanesnet.serialization.config import Config

from .scaling import ScaleFactor


class PolynomialEnvelope(torch.nn.Module):
    """Polynomial envelope that smoothly goes to zero at the cutoff distance.

    Args:
        exponent: Polynomial exponent ``p > 0``.
    """

    def __init__(self, exponent: int) -> None:
        super().__init__()
        assert exponent > 0
        self.p = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Apply the polynomial envelope.

        Args:
            d_scaled: Distances scaled to ``[0, 1]`` by the cutoff radius.

        Returns:
            Envelope values, zero for ``d_scaled >= 1``.
        """
        env_val = 1 + self.a * d_scaled**self.p + self.b * d_scaled ** (self.p + 1) + self.c * d_scaled ** (self.p + 2)
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(torch.nn.Module):
    """Exponential envelope that smoothly goes to zero at the cutoff distance."""

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Apply the exponential envelope.

        Args:
            d_scaled: Distances scaled to ``[0, 1]`` by the cutoff radius.

        Returns:
            Envelope values, zero for ``d_scaled >= 1``.
        """
        env_val = torch.exp(-(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled)))
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class GaussianBasis(torch.nn.Module):
    """Radial Gaussian basis functions with evenly spaced centres.

    Args:
        start: Start of the centre grid in **angstrom**. Default ``0.0``.
        stop: End of the centre grid in **angstrom**. Default ``5.0``.
        num_gaussians: Number of Gaussian functions.
        trainable: Whether the centres are learnable parameters.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        if trainable:
            self.offset = torch.nn.Parameter(offset, requires_grad=True)
        else:
            self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian basis values.

        Args:
            dist: Pairwise distances in **angstrom**, shape ``(nEdges,)``.

        Returns:
            Basis values of shape ``(nEdges, num_gaussians)``.
        """
        dist = dist[:, None] - self.offset[None, :]
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SphericalBesselBasis(torch.nn.Module):
    """Spherical Bessel radial basis functions with trainable frequencies.

    Args:
        num_radial: Number of radial basis functions.
        cutoff: Cutoff radius in **angstrom**.
    """

    def __init__(self, num_radial: int, cutoff: float) -> None:
        super().__init__()
        self.norm_const = math.sqrt(2 / (cutoff**3))
        self.frequencies = torch.nn.Parameter(
            data=torch.tensor(np.pi * np.arange(1, num_radial + 1, dtype=np.float32)),
            requires_grad=True,
        )

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Compute spherical Bessel basis values.

        Args:
            d_scaled: Distances scaled by the cutoff radius, shape
                ``(nEdges,)``.

        Returns:
            Basis values of shape ``(nEdges, num_radial)``.
        """
        return self.norm_const / d_scaled[:, None] * torch.sin(self.frequencies * d_scaled[:, None])


class BernsteinBasis(torch.nn.Module):
    """Bernstein polynomial radial basis functions with trainable decay.

    Args:
        num_radial: Number of Bernstein basis polynomials.
        pregamma_initial: Initial value for the pre-softplus decay parameter.
    """

    def __init__(self, num_radial: int, pregamma_initial: float = 0.45264) -> None:
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer("prefactor", torch.tensor(prefactor, dtype=torch.float), persistent=False)
        self.pregamma = torch.nn.Parameter(data=torch.tensor(pregamma_initial, dtype=torch.float), requires_grad=True)
        self.softplus = torch.nn.Softplus()
        exp1 = torch.arange(num_radial)
        self.register_buffer("exp1", exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer("exp2", exp2[None, :], persistent=False)

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Compute Bernstein basis values.

        Args:
            d_scaled: Distances scaled by the cutoff radius, shape
                ``(nEdges,)``.

        Returns:
            Basis values of shape ``(nEdges, num_radial)``.
        """
        gamma = self.softplus(self.pregamma)
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return self.prefactor * (exp_d**self.exp1) * ((1 - exp_d) ** self.exp2)


class RadialBasis(torch.nn.Module):
    """Combined radial basis with envelope, built from config objects.

    Args:
        num_radial: Number of radial basis functions.
        cutoff: Cutoff radius in **angstrom**.
        rbf: Config specifying the radial basis type and its hyperparameters.
        envelope: Config specifying the envelope type and its hyperparameters.
        scale_basis: If ``True``, apply a learnable :class:`ScaleFactor` after
            the envelope multiplication.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: Config,
        envelope: Config,
        scale_basis: bool = False,
    ) -> None:
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_rbf = ScaleFactor()

        env_name = envelope.get_str("name").lower()
        env_hparams = {k: v for k, v in envelope.as_dict().items() if k != "name"}
        self.envelope: PolynomialEnvelope | ExponentialEnvelope
        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf.get_str("name").lower()
        rbf_hparams = {k: v for k, v in rbf.as_dict().items() if k != "name"}
        self.rbf: GaussianBasis | SphericalBesselBasis | BernsteinBasis
        if rbf_name == "gaussian":
            self.rbf = GaussianBasis(start=0, stop=1, num_gaussians=num_radial, **rbf_hparams)
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(num_radial=num_radial, cutoff=cutoff, **rbf_hparams)
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """Compute enveloped radial basis values.

        Args:
            d: Pairwise distances in **angstrom**, shape ``(nEdges,)``.

        Returns:
            Basis values of shape ``(nEdges, num_radial)``.
        """
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        res = env[:, None] * self.rbf(d_scaled)
        if self.scale_basis:
            res = self.scale_rbf(res)
        return res
