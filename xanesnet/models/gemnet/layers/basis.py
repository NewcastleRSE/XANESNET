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
    """
    1D Bessel Basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int
        Exponent of the envelope function.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int,
    ) -> None:
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
        with torch.no_grad():
            self.frequencies.copy_(
                torch.tensor(np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32))
            )

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        d = d[:, None]  # (nEdges,1)
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        return env * self.norm_const * torch.sin(self.frequencies * d_scaled) / d


class SphericalBasisLayer(torch.nn.Module):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int,
        efficient: bool = False,
    ) -> None:
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

        d_scaled = D_ca * self.inv_cutoff  # (nEdges,)
        u_d = self.envelope(d_scaled)
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        # s: 0 0 0 0 1 1 1 1 ...
        # r: 0 1 2 3 0 1 2 3 ...
        rbf = torch.stack(rbf, dim=1)  # (nEdges, num_spherical * num_radial)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf  # (nEdges, num_spherical * num_radial)

        sph = [f(Angle_cab) for f in self.sph_funcs]
        sph = torch.stack(sph, dim=1)  # (nTriplets, num_spherical)

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
            # maximum number of neighbors, catch empty id_reduce_ji with maximum
            Kmax = 0 if sph.shape[0] == 0 else int(torch.max(torch.max(Kidx + 1), torch.tensor(0)).item())
            nEdges = d_scaled.shape[0]

            sph2 = torch.zeros(nEdges, Kmax, self.num_spherical, device=self.device_buffer.device, dtype=sph.dtype)
            sph2[id3_reduce_ca, Kidx] = sph

            # (num_spherical, nEdges, num_radial), (nEdges, Kmax, num_spherical)
            return rbf_env, sph2


class TensorBasisLayer(torch.nn.Module):
    """
    3D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int,
        efficient=False,
    ) -> None:
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

        d_scaled = D_ca * self.inv_cutoff
        u_d = self.envelope(d_scaled)

        rbf = [f(d_scaled) for f in self.bessel_funcs]
        # s: 0 0 0 0 1 1 1 1 ...
        # r: 0 1 2 3 0 1 2 3 ...
        rbf = torch.stack(rbf, dim=1)  # (nEdges, num_spherical * num_radial)
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

        sph = [f(Alpha_cab, Theta_cabd) for f in self.sph_funcs]
        sph = torch.stack(sph, dim=1)  # (nQuadruplets, num_spherical**2)

        if not self.efficient:
            sph = torch.repeat_interleave(sph, self.num_radial, dim=1)  # (nQuadruplets, num_spherical**2 * num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # Y_lm: l: 0  0    1  1  1  1  1  1    2  2  2  2  2  2  2  2  2  2
            #       m: 0  0   -1 -1  0  0  1  1   -2 -2 -1 -1  0  0  1  1  2  2
            return rbf_env * sph  # (nQuadruplets, num_spherical**2 * num_radial)

        else:
            rbf_env = torch.transpose(rbf_env, 0, 1)  # (num_spherical**2, nEdges, num_radial)

            # Zero padded dense matrix
            # maximum number of neighbors, catch empty id_reduce_ji with maximum
            Kmax = 0 if sph.shape[0] == 0 else int(torch.max(torch.max(Kidx + 1), torch.tensor(0)).item())
            nEdges = d_scaled.shape[0]

            sph2 = torch.zeros(nEdges, Kmax, self.num_spherical**2, device=self.degreeInOrder.device, dtype=sph.dtype)
            sph2[id4_reduce_ca, Kidx] = sph

            # (num_spherical**2, nEdges, num_radial), (nEdges, Kmax, num_spherical**2)
            return rbf_env, sph2


###############################################################################
################################### HELPERS ###################################
###############################################################################


def Jn(r: float, n: int) -> np.floating[Any]:
    """
    numerical spherical bessel functions of order n
    """
    return special.spherical_jn(n, r)


def Jn_zeros(n: int, k: int) -> npt.NDArray[np.float32]:
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
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
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sp.symbols("x")
    # j_i = (-x)^i * (1/x * d/dx)^î * sin(x)/x
    j = [sp.sin(x) / x]  # j_0
    a = sp.sin(x) / x
    for i in range(1, n):
        b = sp.diff(a, x) / x
        j += [sp.simplify(b * (-x) ** i)]
        a = sp.simplify(b)
    return j


def bessel_basis(n: int, k: int) -> list[list[sp.Expr]]:
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).

    Returns:
        bess_basis: list
            Bessel basis formulas taking in a single argument x.
            Has length n where each element has length k. -> In total n*k many.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = (
            1 / np.array(normalizer_tmp) ** 0.5
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
    """
    Computes the constant pre-factor for the spherical harmonic of degree l and order m.

    Parameters
    ----------
        l: int
            Degree of the spherical harmonic. l >= 0
        m: int
            Order of the spherical harmonic. -l <= m <= l

    Returns
    -------
        factor: float

    """
    # sqrt((2*l+1)/4*pi * (l-m)!/(l+m)! )
    return ((2 * l + 1) / (4 * np.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m))) ** 0.5


def associated_legendre_polynomials(L: int, zero_m_only: bool = True, pos_m_only: bool = True) -> list[list[Any]]:
    """
    Computes string formulas of the associated legendre polynomials up to degree L (excluded).

    Parameters
    ----------
        L: int
            Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
        zero_m_only: bool
            If True only calculate the polynomials for the polynomials where m=0.
        pos_m_only: bool
            If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.

    Returns
    -------
        polynomials: list
            Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).
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
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.

    Parameters
    ----------
        L: int
            Degree up to which to calculate the spherical harmonics (degree L is excluded).
        spherical_coordinates: bool
            - True: Expects the input of the formula strings to be phi and theta.
            - False: Expects the input of the formula strings to be x, y and z.
        zero_m_only: bool
            If True only calculate the harmonics where m=0.

    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
            the total count is reduced to be only L many.
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
