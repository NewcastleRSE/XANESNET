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

"""Spherical Bessel and harmonic basis utilities for GemNet-OC."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import sympy as sym
import torch
from scipy import special as sp
from scipy.optimize import brentq


def Jn(r: float, n: int) -> np.floating[Any]:
    """Numerical spherical Bessel function of order ``n``.

    Args:
        r: Evaluation point.
        n: Order of the spherical Bessel function.

    Returns:
        Value of the ``n``-th spherical Bessel function at ``r``.
    """
    return sp.spherical_jn(n, r)


def Jn_zeros(n: int, k: int) -> npt.NDArray[np.float32]:
    """Compute the first ``k`` zeros of the spherical Bessel functions up to order ``n``.

    Args:
        n: Compute zeros for orders ``0, 1, ..., n - 1``.
        k: Number of zeros per order.

    Returns:
        Array of zeros of shape ``(n, k)``.
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            racines[j] = brentq(Jn, points[j], points[j + 1], (i,))
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n: int) -> list[sym.Expr]:
    """Compute sympy formulas for the spherical Bessel functions up to order ``n``.

    Args:
        n: Compute formulas for orders ``0, 1, ..., n - 1``.

    Returns:
        List of ``n`` sympy expressions in the variable ``x``.
    """
    x = sym.symbols("x", real=True)
    # j_i = (-x)^i * (1/x * d/dx)^î * sin(x)/x
    j = [sym.sin(x) / x]  # j_0
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n: int, k: int) -> list[list[sym.Expr]]:
    """Compute sympy formulas for the normalised rescaled spherical Bessel basis.

    Args:
        n: Maximum order (excluded); produces formulas for orders
            ``0, 1, ..., n - 1``.
        k: Number of frequency components per order.

    Returns:
        Nested list of shape ``(n, k)`` containing sympy expressions that
        each take a single argument ``x``.
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
    x = sym.symbols("x", real=True)
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order][i] * f[order].subs(x, zeros[order, i] * x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l_degree: int, m_order: int) -> float:
    """Compute the normalisation pre-factor for a real spherical harmonic.

    Computes ``sqrt((2l+1) / (4 pi) * (l-|m|)! / (l+|m|)!)``.

    Args:
        l_degree: Degree of the spherical harmonic (``l >= 0``).
        m_order: Order of the spherical harmonic (``-l <= m <= l``).

    Returns:
        Normalisation pre-factor as a Python float.
    """
    # sqrt((2*l+1)/4*pi * (l-m)!/(l+m)! )
    return (
        (2 * l_degree + 1)
        / (4 * np.pi)
        * math.factorial(l_degree - abs(m_order))
        / math.factorial(l_degree + abs(m_order))
    ) ** 0.5


def associated_legendre_polynomials(
    L_maxdegree: int, zero_m_only: bool = True, pos_m_only: bool = True
) -> list[list[Any]]:
    """Compute sympy formulas for the associated Legendre polynomials.

    Args:
        L_maxdegree: Degree upper bound (excluded); compute polynomials for
            degrees ``0, 1, ..., L_maxdegree - 1``.
        zero_m_only: If ``True``, only compute polynomials for ``m = 0``.
        pos_m_only: If ``True``, only compute polynomials for ``m >= 0``.
            Ignored when ``zero_m_only`` is ``True``.

    Returns:
        Nested list of sympy expressions indexed as ``P[l][m]``.
    """
    # calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html
    z = sym.symbols("z", real=True)
    P_l_m = [[0] * (2 * l_degree + 1) for l_degree in range(L_maxdegree)]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L_maxdegree > 1:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l_degree in range(2, L_maxdegree):
                P_l_m[l_degree][0] = sym.simplify(
                    ((2 * l_degree - 1) * z * P_l_m[l_degree - 1][0] - (l_degree - 1) * P_l_m[l_degree - 2][0])
                    / l_degree
                )
            return P_l_m
        else:
            # for m >= 0
            for l_degree in range(1, L_maxdegree):
                P_l_m[l_degree][l_degree] = sym.simplify(
                    (1 - 2 * l_degree) * (1 - z**2) ** 0.5 * P_l_m[l_degree - 1][l_degree - 1]
                )  # P_00, P_11, P_22, P_33

            for m_order in range(L_maxdegree - 1):
                P_l_m[m_order + 1][m_order] = sym.simplify(
                    (2 * m_order + 1) * z * P_l_m[m_order][m_order]
                )  # P_10, P_21, P_32, P_43

            for l_degree in range(2, L_maxdegree):
                for m_order in range(l_degree - 1):  # P_20, P_30, P_31
                    P_l_m[l_degree][m_order] = sym.simplify(
                        (
                            (2 * l_degree - 1) * z * P_l_m[l_degree - 1][m_order]
                            - (l_degree + m_order - 1) * P_l_m[l_degree - 2][m_order]
                        )
                        / (l_degree - m_order)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l_degree in range(1, L_maxdegree):
                    for m_order in range(1, l_degree + 1):  # P_1(-1), P_2(-1) P_2(-2)
                        P_l_m[l_degree][-m_order] = sym.simplify(
                            (-1) ** m_order
                            * math.factorial(l_degree - m_order)
                            / math.factorial(l_degree + m_order)
                            * P_l_m[l_degree][m_order]
                        )

                return P_l_m
            return P_l_m

    return P_l_m


def real_sph_harm(
    L_maxdegree: int,
    use_theta: bool,
    use_phi: bool = True,
    zero_m_only: bool = True,
) -> list[list[Any]]:
    """Compute sympy formulas for the real spherical harmonics up to degree ``L``.

    Variables are spherical coordinates (``phi``, ``theta``) or Cartesian
    coordinates (``x``, ``y``, ``z``) on the unit sphere.

    Args:
        L_maxdegree: Degree upper bound (excluded).
        use_theta: If ``True``, replace ``z`` by ``cos(theta)`` in the
            expressions. If ``False``, use ``z`` directly.
        use_phi: If ``True``, keep ``phi`` in the expressions. If ``False``,
            replace ``phi`` by ``atan2(y, x)``. Ignored when
            ``zero_m_only=True``.
        zero_m_only: If ``True``, only compute harmonics for ``m = 0``.

    Returns:
        Nested list of sympy expressions ``Y[l][m]``, containing ``L``
        elements when ``zero_m_only=True`` or ``L**2`` elements otherwise.
    """
    z = sym.symbols("z", real=True)
    P_l_m = associated_legendre_polynomials(L_maxdegree, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l_degree in range(L_maxdegree)]
    else:
        Y_l_m = [[0] * (2 * l_degree + 1) for l_degree in range(L_maxdegree)]  # for order l: -l <= m <= l

    # convert expressions to spherical coordiantes
    if use_theta:
        # replace z by cos(theta)
        theta = sym.symbols("theta", real=True)
        for l_degree in range(L_maxdegree):
            for m_order in range(len(P_l_m[l_degree])):
                if not isinstance(P_l_m[l_degree][m_order], int):
                    P_l_m[l_degree][m_order] = P_l_m[l_degree][m_order].subs(z, sym.cos(theta))  # type: ignore[union-attr]

    ## calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    #             { sqrt(2) * (-1)^m * N * P_l|m| * sin(|m|*phi)   if m < 0
    # Y_lm_real = { Y_lm                                           if m = 0
    #             { sqrt(2) * (-1)^m * N * P_lm * cos(m*phi)       if m > 0

    for l_degree in range(L_maxdegree):
        Y_l_m[l_degree][0] = sym.simplify(sph_harm_prefactor(l_degree, 0) * P_l_m[l_degree][0])  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi", real=True)
        for l_degree in range(1, L_maxdegree):
            # m > 0
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][m_order] = sym.simplify(
                    2**0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.cos(m_order * phi)
                )
            # m < 0
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][-m_order] = sym.simplify(
                    2**0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, -m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.sin(m_order * phi)
                )

        # convert expressions to cartesian coordinates
        if not use_phi:
            # replace phi by atan2(y,x)
            x, y = sym.symbols("x y", real=True)
            for l_degree in range(L_maxdegree):
                for m_order in range(len(Y_l_m[l_degree])):
                    Y_l_m[l_degree][m_order] = sym.simplify(Y_l_m[l_degree][m_order].subs(phi, sym.atan2(y, x)))  # type: ignore[union-attr, attr-defined]
    return Y_l_m


def get_sph_harm_basis(L_maxdegree: int, zero_m_only: bool = True):
    """Return a callable that computes the spherical-harmonic basis from ``z`` and ``phi``.

    Args:
        L_maxdegree: Maximum spherical-harmonic degree (excluded).
        zero_m_only: If ``True``, only include ``m = 0`` harmonics.

    Returns:
        A function ``basis_fn(*args) -> torch.Tensor`` where ``args`` is
        either ``[cos_phi]`` (when ``zero_m_only=True``) or
        ``[cos_phi, theta]``. Output shape is ``(N, num_harmonics)``.
    """
    # retrieve equations
    Y_lm = real_sph_harm(L_maxdegree, use_theta=False, use_phi=True, zero_m_only=zero_m_only)
    Y_lm_flat = [Y for Y_l in Y_lm for Y in Y_l]

    # convert to pytorch functions
    z = sym.symbols("z", real=True)
    variables = [z]
    if not zero_m_only:
        variables.append(sym.symbols("phi", real=True))

    modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
    sph_funcs = sym.lambdify(variables, Y_lm_flat, modules)

    # Return as a single function
    # args are either [cosφ] or [cosφ, ϑ]
    def basis_fn(*args) -> torch.Tensor:
        basis = sph_funcs(*args)
        basis[0] = args[0].new_tensor(basis[0]).expand_as(args[0])
        return torch.stack(basis, dim=1)

    return basis_fn
