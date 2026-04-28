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

"""Gaussian spectral basis and coefficient-fitting utilities for XANESNET."""

import math

import torch
import torch.nn as nn


class SpectralBasis(nn.Module):
    """Gaussian spectral basis for expanding and reconstructing XANES spectra.

    Builds a ``(N, K)`` basis matrix ``Phi`` on a uniform energy grid, where
    ``N`` is the number of energy points and
    ``K = len(widths_eV) * ceil(N / stride)``. Each column of ``Phi`` is a
    Gaussian centred at a strided grid point with one of the specified widths.
    Both ``Phi`` and the ``centers`` vector are registered as non-trainable
    buffers.

    Args:
        energies: ``(N,)`` float -- uniformly spaced energy grid in
            **eV**.
        widths_eV: Gaussian standard deviations in **eV** for each basis
            family. The total number of basis functions is
            ``len(widths_eV) * ceil(N / stride)``.
        normalize_atoms: If ``True`` (default), each Gaussian column is
            normalized so that its discrete integral (sum * dE) is
            approximately 1.
        stride: Spacing between Gaussian centres in grid steps. A stride
            of 1 places a centre at every energy point.
    """

    def __init__(
        self,
        energies: torch.Tensor,
        widths_eV: list[float],
        normalize_atoms: bool = True,
        stride: int = 1,
    ) -> None:
        """Initialize the Gaussian spectral basis."""
        super().__init__()

        self.register_buffer("E", energies.detach().clone())
        self.widths_eV = widths_eV
        self.normalize_atoms = bool(normalize_atoms)
        self.stride = int(stride)

        N = energies.numel()
        dE = float(energies[1] - energies[0])

        widths_bins: list[float] = [max(w / dE, 0.5) for w in self.widths_eV]

        grid_idx = torch.arange(N, device=energies.device, dtype=energies.dtype)
        centers_grid = grid_idx[:: self.stride]
        diff_bins = grid_idx.unsqueeze(1) - centers_grid.unsqueeze(0)

        Phi_list, centers_list = [], []
        for w in widths_bins:
            Phi_w = torch.exp(-0.5 * (diff_bins / w) ** 2)
            Phi_list.append(Phi_w)
            centers_list.append(self.E[:: self.stride])

        Phi = torch.cat(Phi_list, dim=1)
        centers = torch.cat(centers_list)

        if self.normalize_atoms:
            Phi = Phi / (Phi.sum(dim=0, keepdim=True) * dE + 1e-12)

        self.register_buffer("Phi", Phi)
        self.register_buffer("centers", centers)

    def synthesize(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Reconstruct spectra from Gaussian basis coefficients.

        Args:
            coeffs: ``(*, K)`` float -- basis coefficients.

        Returns:
            ``(*, N)`` float -- synthesised spectra.
        """
        return coeffs @ self.Phi.T


class SpectralPost(nn.Module):
    """Parameter-free synthesis module: reconstructs spectra from coefficients.

    Computes ``y = Phi @ c`` (optionally clamped to non-negative values).
    Contains no learnable parameters; intended as a post-processing stage
    after a coefficient-predicting network.

    Args:
        basis: Pre-built ``SpectralBasis`` whose ``Phi`` matrix is used
            for synthesis.
        nonneg_output: If ``True``, synthesised spectra are clamped to
            non-negative values.
    """

    def __init__(
        self,
        basis: "SpectralBasis",
        nonneg_output: bool = False,
    ) -> None:
        """Initialize the synthesis module."""
        super().__init__()
        self.basis = basis
        self.nonneg_output = bool(nonneg_output)

    def forward_from_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for :meth:`forward`.

        Args:
            coeffs: ``(*, K)`` float -- Gaussian basis coefficients.

        Returns:
            ``(*, N)`` float -- synthesised spectra, optionally clamped to
            non-negative values.
        """
        return self.forward(coeffs)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Synthesise spectra from Gaussian basis coefficients.

        Args:
            coeffs: ``(*, K)`` float -- Gaussian basis coefficients.

        Returns:
            ``(*, N)`` float -- synthesised spectra, optionally clamped to
            non-negative values.
        """
        y = self.basis.synthesize(coeffs)
        if self.nonneg_output:
            y = y.clamp_min_(0)
        return y


def build_ridge_operator(phi: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """Build the Tikhonov-regularised least-squares projection operator.

    Computes ``A = (Phi.T @ Phi + lam * I)^{-1} @ Phi.T`` via Cholesky
    decomposition, with a fallback to augmented least-squares when the Gram
    matrix is not positive-definite.

    Args:
        phi: ``(N_E, K)`` float -- basis matrix.
        lam: Tikhonov regularisation strength (L2 penalty on coefficients).

    Returns:
        ``(K, N_E)`` float32 projection operator on the same device as
        ``phi``.
    """
    phi = phi.contiguous()
    N_E, K = phi.shape
    I_K = torch.eye(K, dtype=phi.dtype, device=phi.device)

    G = phi.T @ phi
    G = G + lam * I_K
    try:
        L = torch.linalg.cholesky(G)  # (K,K)
        A = torch.cholesky_solve(phi.T, L)  # (K,N_E)
    except RuntimeError:
        top = phi
        bot = math.sqrt(lam) * I_K
        A_aug = torch.cat([top, bot], dim=0)  # ((N_E+K), K)
        rhs = torch.cat(
            [
                torch.eye(N_E, dtype=phi.dtype, device=phi.device),
                torch.zeros((K, N_E), dtype=phi.dtype, device=phi.device),
            ],
            dim=0,
        )  # ((N_E+K), N_E)
        A = torch.linalg.lstsq(A_aug, rhs, rcond=None).solution  # (K, N_E)

    return A.to(torch.float32)


def gaussian_fit(basis: SpectralBasis, xanes: torch.Tensor) -> torch.Tensor:
    """Fit Gaussian basis coefficients to observed XANES spectra.

    Solves the Tikhonov-regularised least-squares problem
    ``argmin_c ||Phi c - y||^2 + 1e-2 * ||c||^2`` for each row in ``xanes``.

    Args:
        basis: Pre-built ``SpectralBasis`` instance.
        xanes: ``(*, N_E)`` float -- observed spectra.

    Returns:
        ``(*, K)`` float -- Gaussian basis coefficients.
    """
    A = build_ridge_operator(basis.Phi, lam=1e-2)

    return xanes @ A.T


def gaussian_inverse(
    basis: SpectralBasis,
    coeffs: torch.Tensor,
    nonneg_output: bool = False,
) -> torch.Tensor:
    """Reconstruct spectra from Gaussian basis coefficients.

    Args:
        basis: Pre-built ``SpectralBasis`` instance.
        coeffs: ``(*, K)`` float -- Gaussian basis coefficients.
        nonneg_output: If ``True``, reconstructed spectra are clamped to
            non-negative values.

    Returns:
        ``(*, N_E)`` float -- reconstructed spectra.
    """
    y = basis.synthesize(coeffs)
    if nonneg_output:
        y = y.clamp_min_(0)
    return y
