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

"""Bessel, spherical, and tensor basis layers with associated mathematical utilities."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import sympy as sp
import torch
from scipy import special
from scipy.optimize import brentq

from .envelope import Envelope


class BesselBasisLayer(torch.nn.Module):
    """1D Bessel radial basis functions with smooth cutoff envelope.

    Args:
        num_radial: Number of basis functions (controls maximum frequency).
        cutoff: Radial cutoff distance in **A**.
        envelope_exponent: Exponent ``p`` of the polynomial cutoff envelope.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int,
    ) -> None:
        """Initialize ``BesselBasisLayer``."""
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5

        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.Tensor(np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32)),
            requires_grad=True,
        )

    def reset_parameters(self) -> None:
        """Re-initialize frequencies to their canonical values ``pi * [1, ..., num_radial]``."""
        with torch.no_grad():
            self.frequencies.copy_(torch.tensor(np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32)))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """Evaluate the Bessel basis on a set of edge distances.

        Args:
            d: Edge distances in **A**, shape ``(nEdges,)``.

        Returns:
            Basis features of shape ``(nEdges, num_radial)``.
        """
        d = d[:, None]  # (nEdges,1)
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        return env * self.norm_const * torch.sin(self.frequencies * d_scaled) / d


class SphericalBasisLayer(torch.nn.Module):
    """2D Fourier-Bessel basis (radial x spherical harmonic, single angle).

    Args:
        num_spherical: Number of spherical harmonics (controls angular frequency).
        num_radial: Number of radial basis functions per spherical harmonic.
        cutoff: Radial cutoff distance in **A**.
        envelope_exponent: Exponent ``p`` of the polynomial cutoff envelope.
        efficient: If ``True``, use the memory-efficient sparse-dense formulation
            (returns a tuple); otherwise returns a dense tensor.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int,
        efficient: bool = False,
    ) -> None:
        """Initialize ``SphericalBasisLayer``."""
        super().__init__()

        assert num_radial <= 64
        self.efficient = efficient
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.envelope = Envelope(envelope_exponent)
        self.inv_cutoff = 1 / cutoff

        # retrieve formulas
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(num_spherical, spherical_coordinates=True, zero_m_only=True)
        self.sph_funcs = []  # (num_spherical,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)
        self.norm_const = self.inv_cutoff**1.5
        self.register_buffer("device_buffer", torch.zeros(0), persistent=False)  # dummy buffer to get device of layer

        # convert to torch functions
        x = sp.symbols("x")
        theta = sp.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        m = 0  # only single angle
        for l in range(len(Y_lm)):  # num_spherical
            if l == 0:
                # Y_00 is only a constant -> function returns value and not tensor
                first_sph = sp.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(lambda theta: torch.zeros_like(theta) + first_sph(theta))
            else:
                self.sph_funcs.append(sp.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(sp.lambdify([x], bessel_formulas[l][n], modules))

    def forward(
        self,
        D_ca: torch.Tensor,
        Angle_cab: torch.Tensor,
        id3_reduce_ca: torch.Tensor,
        Kidx: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the 2D spherical basis on triplet geometry.

        Args:
            D_ca: Edge distances c -> a in **A**, shape ``(nEdges,)``.
            Angle_cab: Triplet angles at atom a (c-a-b), shape ``(nTriplets,)``.
            id3_reduce_ca: Edge index c -> a for each triplet, shape ``(nTriplets,)``.
            Kidx: Neighbor index within the sparse dense matrix, shape ``(nTriplets,)``.

        Returns:
            If ``efficient=False``: dense basis of shape
            ``(nTriplets, num_spherical * num_radial)``.

            If ``efficient=True``: tuple ``(rbf_env, sph2)`` where
            ``rbf_env`` has shape ``(num_spherical, nEdges, num_radial)``
            and ``sph2`` has shape ``(nEdges, Kmax, num_spherical)``.
        """
        d_scaled = D_ca * self.inv_cutoff  # (nEdges,)
        u_d = self.envelope(d_scaled)
        # s: 0 0 0 0 1 1 1 1 ...
        # r: 0 1 2 3 0 1 2 3 ...
        rbf: torch.Tensor = torch.stack(
            [f(d_scaled) for f in self.bessel_funcs], dim=1
        )  # (nEdges, num_spherical * num_radial)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf  # (nEdges, num_spherical * num_radial)

        sph: torch.Tensor = torch.stack([f(Angle_cab) for f in self.sph_funcs], dim=1)  # (nTriplets, num_spherical)

        if not self.efficient:
            rbf_env = rbf_env[id3_reduce_ca]  # (nTriplets, num_spherical * num_radial)
            rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # z_ln: l: 0 0  1 1  2 2
            #       n: 0 1  0 1  0 1
            sph = sph.view(-1, self.num_spherical, 1)  # (nTriplets, num_spherical, 1)
            # e.g. num_spherical = 3, num_radial = 2
            # Y_lm: l: 0 0  1 1  2 2
            #       m: 0 0  0 0  0 0
            out = (rbf_env * sph).view(-1, self.num_spherical * self.num_radial)
            return out  # (nTriplets, num_spherical * num_radial)
        else:
            rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            rbf_env = torch.transpose(rbf_env, 0, 1)  # (num_spherical, nEdges, num_radial)

            # Zero padded dense matrix
            # Maximum number of neighbors; keep this device-safe for CUDA tensors.
            Kmax = int(Kidx.max().item()) + 1 if Kidx.numel() > 0 else 0
            nEdges = d_scaled.shape[0]

            sph2 = torch.zeros(nEdges, Kmax, self.num_spherical, device=self.device_buffer.device, dtype=sph.dtype)
            sph2[id3_reduce_ca, Kidx] = sph

            # (num_spherical, nEdges, num_radial), (nEdges, Kmax, num_spherical)
            return rbf_env, sph2


class TensorBasisLayer(torch.nn.Module):
    """3D Fourier-Bessel basis (radial x two spherical harmonics, two angles).

    Args:
        num_spherical: Number of spherical harmonics (controls angular frequency).
        num_radial: Number of radial basis functions per spherical harmonic.
        cutoff: Radial cutoff distance in **A**.
        envelope_exponent: Exponent ``p`` of the polynomial cutoff envelope.
        efficient: If ``True``, use the memory-efficient sparse-dense formulation
            (returns a tuple); otherwise returns a dense tensor.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int,
        efficient: bool = False,
    ) -> None:
        """Initialize ``TensorBasisLayer``."""
        super().__init__()

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.efficient = efficient

        self.inv_cutoff = 1 / cutoff
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(num_spherical, spherical_coordinates=True, zero_m_only=False)
        self.sph_funcs = []  # (num_spherical**2,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)
        self.norm_const = self.inv_cutoff**1.5

        # convert to torch functions
        x = sp.symbols("x")
        theta = sp.symbols("theta")
        phi = sp.symbols("phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        for l in range(len(Y_lm)):  # num_spherical
            for m in range(len(Y_lm[l])):
                if l == 0:  # Y_00 is only a constant -> function returns value and not tensor
                    first_sph = sp.lambdify([theta, phi], Y_lm[l][m], modules)
                    self.sph_funcs.append(lambda theta, phi: torch.zeros_like(theta) + first_sph(theta, phi))
                else:
                    self.sph_funcs.append(sp.lambdify([theta, phi], Y_lm[l][m], modules))
            for j in range(num_radial):
                self.bessel_funcs.append(sp.lambdify([x], bessel_formulas[l][j], modules))

        self.register_buffer("degreeInOrder", torch.arange(num_spherical) * 2 + 1, persistent=False)

    def forward(
        self,
        D_ca: torch.Tensor,
        Alpha_cab: torch.Tensor,
        Theta_cabd: torch.Tensor,
        id4_reduce_ca: torch.Tensor,
        Kidx: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the 3D spherical basis on quadruplet geometry.

        Args:
            D_ca: Edge distances c -> a in **A**, shape ``(nEdges,)``.
            Alpha_cab: First angle (c-a-b) for each quadruplet, shape
                ``(nQuadruplets,)``.
            Theta_cabd: Second angle (dihedral-like c-a-b-d), shape
                ``(nQuadruplets,)``.
            id4_reduce_ca: Quadruplet reduce index c -> a, shape
                ``(nQuadruplets,)``.
            Kidx: Neighbor index within the sparse dense matrix, shape
                ``(nQuadruplets,)``.

        Returns:
            If ``efficient=False``: dense basis of shape
            ``(nQuadruplets, num_spherical**2 * num_radial)``.

            If ``efficient=True``: tuple ``(rbf_env, sph2)`` where
            ``rbf_env`` has shape ``(num_spherical**2, nEdges, num_radial)``
            and ``sph2`` has shape ``(nEdges, Kmax, num_spherical**2)``.
        """
        d_scaled = D_ca * self.inv_cutoff
        u_d = self.envelope(d_scaled)

        # s: 0 0 0 0 1 1 1 1 ...
        # r: 0 1 2 3 0 1 2 3 ...
        rbf: torch.Tensor = torch.stack(
            [f(d_scaled) for f in self.bessel_funcs], dim=1
        )  # (nEdges, num_spherical * num_radial)
        rbf = rbf * self.norm_const

        rbf_env = u_d[:, None] * rbf  # (nEdges, num_spherical * num_radial)
        rbf_env = rbf_env.view((-1, self.num_spherical, self.num_radial))  # (nEdges, num_spherical, num_radial)
        rbf_env = torch.repeat_interleave(rbf_env, self.degreeInOrder, dim=1)  # (nEdges, num_spherical**2, num_radial)

        if not self.efficient:
            rbf_env = rbf_env.view(
                (-1, self.num_spherical**2 * self.num_radial)
            )  # (nEdges, num_spherical**2 * num_radial)
            rbf_env = rbf_env[id4_reduce_ca]  # (nQuadruplets, num_spherical**2 * num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # j_ln: l: 0  0    1  1  1  1  1  1    2  2  2  2  2  2  2  2  2  2
            #       n: 0  1    0  1  0  1  0  1    0  1  0  1  0  1  0  1  0  1

        sph: torch.Tensor = torch.stack(
            [f(Alpha_cab, Theta_cabd) for f in self.sph_funcs], dim=1
        )  # (nQuadruplets, num_spherical**2)

        if not self.efficient:
            sph = torch.repeat_interleave(sph, self.num_radial, dim=1)  # (nQuadruplets, num_spherical**2 * num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # Y_lm: l: 0  0    1  1  1  1  1  1    2  2  2  2  2  2  2  2  2  2
            #       m: 0  0   -1 -1  0  0  1  1   -2 -2 -1 -1  0  0  1  1  2  2
            return rbf_env * sph  # (nQuadruplets, num_spherical**2 * num_radial)

        else:
            rbf_env = torch.transpose(rbf_env, 0, 1)  # (num_spherical**2, nEdges, num_radial)

            # Zero padded dense matrix
            # Maximum number of neighbors; keep this device-safe for CUDA tensors.
            Kmax = int(Kidx.max().item()) + 1 if Kidx.numel() > 0 else 0
            nEdges = d_scaled.shape[0]

            sph2 = torch.zeros(nEdges, Kmax, self.num_spherical**2, device=self.degreeInOrder.device, dtype=sph.dtype)
            sph2[id4_reduce_ca, Kidx] = sph

            # (num_spherical**2, nEdges, num_radial), (nEdges, Kmax, num_spherical**2)
            return rbf_env, sph2


###############################################################################
################################### HELPERS ###################################
###############################################################################


def Jn(r: float, n: int) -> np.floating[Any]:
    """Evaluate the spherical Bessel function of order ``n`` at ``r``.

    Args:
        r: Evaluation point.
        n: Order of the spherical Bessel function. Non-negative integer.

    Returns:
        Value of ``j_n(r)``.
    """
    return special.spherical_jn(n, r)


def Jn_zeros(n: int, k: int) -> npt.NDArray[np.float32]:
    """Compute the first ``k`` zeros of the spherical Bessel functions up to order ``n``.

    Args:
        n: Maximum order (exclusive). Zeros for orders 0, 1, ..., n-1 are computed.
        k: Number of zeros to compute per order.

    Returns:
        Array of shape ``(n, k)`` containing the zeros.
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n: int) -> list[sp.Expr]:
    """Compute sympy formulas for the spherical Bessel functions up to order ``n``.

    Args:
        n: Maximum order (exclusive). Formulas for orders 0, 1, ..., n-1 are returned.

    Returns:
        List of ``n`` sympy expressions in ``x``.
    """
    x = sp.symbols("x")
    # j_i = (-x)^i * (1/x * d/dx)^i * sin(x)/x
    j = [sp.sin(x) / x]  # j_0
    a = sp.sin(x) / x
    for i in range(1, n):
        b = sp.diff(a, x) / x
        j += [sp.simplify(b * (-x) ** i)]
        a = sp.simplify(b)
    return j


def bessel_basis(n: int, k: int) -> list[list[sp.Expr]]:
    """Compute sympy formulas for the normalized, rescaled spherical Bessel basis.

    Args:
        n: Maximum order (exclusive).
        k: Number of basis functions per order (maximum frequency, exclusive).

    Returns:
        Nested list of shape ``(n, k)`` containing sympy expressions in ``x``
        representing the normalized basis functions.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_list = []
        for i in range(k):
            normalizer_list += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = (
            1 / np.array(normalizer_list) ** 0.5
        )  # sqrt(2/(j_l+1)**2) , sqrt(1/c**3) not taken into account yet
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sp.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sp.simplify(normalizer[order][i] * f[order].subs(x, zeros[order, i] * x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l: int, m: int) -> float:
    """Compute the constant pre-factor for real spherical harmonic ``Y_l^m``.

    Pre-factor: ``sqrt((2l+1) / (4*pi) * (l-|m|)! / (l+|m|)!)``.

    Args:
        l: Degree of the spherical harmonic. Must satisfy ``l >= 0``.
        m: Order of the spherical harmonic. Must satisfy ``-l <= m <= l``.

    Returns:
        Pre-factor value.
    """
    # sqrt((2l+1) / (4*pi) * (l-|m|)! / (l+|m|)!)
    return ((2 * l + 1) / (4 * np.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m))) ** 0.5


def associated_legendre_polynomials(L: int, zero_m_only: bool = True, pos_m_only: bool = True) -> list[list[Any]]:
    """Compute sympy formulas for the associated Legendre polynomials up to degree ``L``.

    Args:
        L: Maximum degree (exclusive). Polynomials for degrees 0, 1, ..., L-1.
        zero_m_only: If ``True``, compute only the ``m = 0`` polynomials.
        pos_m_only: If ``True``, compute only the ``m >= 0`` polynomials.
            Overridden by ``zero_m_only``.

    Returns:
        Nested list where ``result[l]`` has length ``2*l + 1`` and
        ``result[l][m]`` is the sympy formula for ``P_l^m``.
        Entries are ``0`` (integer) for uncomputed orders.
    """
    # calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html
    z = sp.symbols("z")
    P_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L > 0:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l in range(2, L):
                P_l_m[l][0] = sp.simplify(((2 * l - 1) * z * P_l_m[l - 1][0] - (l - 1) * P_l_m[l - 2][0]) / l)
        else:
            # for m >= 0
            for l in range(1, L):
                P_l_m[l][l] = sp.simplify(
                    (1 - 2 * l) * (1 - z**2) ** 0.5 * P_l_m[l - 1][l - 1]
                )  # P_00, P_11, P_22, P_33

            for m in range(0, L - 1):
                P_l_m[m + 1][m] = sp.simplify((2 * m + 1) * z * P_l_m[m][m])  # P_10, P_21, P_32, P_43

            for l in range(2, L):
                for m in range(l - 1):  # P_20, P_30, P_31
                    P_l_m[l][m] = sp.simplify(
                        ((2 * l - 1) * z * P_l_m[l - 1][m] - (l + m - 1) * P_l_m[l - 2][m]) / (l - m)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l in range(1, L):
                    for m in range(1, l + 1):  # P_1(-1), P_2(-1) P_2(-2)
                        P_l_m[l][-m] = sp.simplify(
                            (-1) ** m * math.factorial(l - m) / math.factorial(l + m) * P_l_m[l][m]
                        )

    return P_l_m


def real_sph_harm(L: int, spherical_coordinates: bool, zero_m_only: bool = True) -> list[list[Any]]:
    """Compute sympy formulas for the real spherical harmonics up to degree ``L``.

    The variables are either spherical coordinates (``phi``, ``theta``) or
    Cartesian coordinates (``x``, ``y``, ``z``) on the unit sphere.

    Args:
        L: Maximum degree (exclusive). Harmonics for degrees 0, 1, ..., L-1.
        spherical_coordinates: If ``True``, formulas use ``phi`` and ``theta``.
            If ``False``, formulas use ``x``, ``y``, and ``z``.
        zero_m_only: If ``True``, compute only the ``m = 0`` harmonics.

    Returns:
        Nested list where ``result[l]`` has length ``1`` if ``zero_m_only``
        is ``True``, otherwise length ``2*l + 1``. Each entry is a sympy
        expression or ``0`` for uncomputed harmonics.
    """
    z = sp.symbols("z")
    P_l_m = associated_legendre_polynomials(L, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l in range(L)]
    else:
        Y_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    # convert expressions to spherical coordiantes
    if spherical_coordinates:
        # replace z by cos(theta)
        theta = sp.symbols("theta")
        for l in range(L):
            for m in range(len(P_l_m[l])):
                if not isinstance(P_l_m[l][m], int):
                    P_l_m[l][m] = P_l_m[l][m].subs(z, sp.cos(theta))

    for l in range(L):
        Y_l_m[l][0] = sp.simplify(sph_harm_prefactor(l, 0) * P_l_m[l][0])  # Y_l0

    if not zero_m_only:
        phi = sp.symbols("phi")
        for l in range(1, L):
            # m > 0
            for m in range(1, l + 1):
                Y_l_m[l][m] = sp.simplify(2**0.5 * (-1) ** m * sph_harm_prefactor(l, m) * P_l_m[l][m] * sp.cos(m * phi))
            # m < 0
            for m in range(1, l + 1):
                Y_l_m[l][-m] = sp.simplify(
                    2**0.5 * (-1) ** m * sph_harm_prefactor(l, -m) * P_l_m[l][m] * sp.sin(m * phi)
                )

        # convert expressions to cartesian coordinates
        if not spherical_coordinates:
            # replace phi by atan2(y,x)
            x = sp.symbols("x")
            y = sp.symbols("y")
            for l in range(L):
                for m in range(len(Y_l_m[l])):
                    val = Y_l_m[l][m]
                    if not isinstance(val, int):
                        Y_l_m[l][m] = sp.simplify(val.subs(phi, sp.atan2(y, x)))
    return Y_l_m
