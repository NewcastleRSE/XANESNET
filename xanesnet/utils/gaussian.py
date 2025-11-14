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
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class SpectralBasis(nn.Module):
    def __init__(
        self,
        energies: torch.Tensor,
        widths_eV: List,
        normalize_atoms=True,
        stride=1,
    ):
        super().__init__()

        self.register_buffer("E", energies.detach().clone())
        self.widths_eV = widths_eV
        self.normalize_atoms = bool(normalize_atoms)
        self.stride = int(stride)

        N = energies.numel()
        dE = float(energies[1] - energies[0])

        widths_bins = tuple(max(w / dE, 0.5) for w in self.widths_eV)
        widths_bins = [float(w) for w in widths_bins]

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

    def synthesize(self, coeffs: torch.Tensor):
        return coeffs @ self.Phi.T


class SpectralPost(nn.Module):
    """
    Stage 1: parameter-free (no training).
      y = Φ c    (optionally clamped to nonnegative)
    """

    def __init__(self, basis: "SpectralBasis", nonneg_output: bool = False):
        super().__init__()
        self.basis = basis
        self.nonneg_output = bool(nonneg_output)

    def forward_from_coeffs(self, coeffs: torch.Tensor):
        y = self.basis.synthesize(coeffs)
        if self.nonneg_output:
            y = y.clamp_min_(0)
        return y


def build_ridge_operator(Phi: Tensor, lam: float = 1e-2) -> Tensor:
    """
    A = (Φᵀ Φ + λ I)^{-1} Φᵀ  with Cholesky; fallback to augmented LSQ.
    Returns A: (K, N_E) on same device/dtype as Phi.
    """
    Phi = Phi.contiguous()
    N_E, K = Phi.shape
    I_K = torch.eye(K, dtype=Phi.dtype, device=Phi.device)

    G = Phi.T @ Phi
    G = G + lam * I_K
    try:
        L = torch.linalg.cholesky(G)  # (K,K)
        A = torch.cholesky_solve(Phi.T, L)  # (K,N_E)
    except RuntimeError:
        top = Phi
        bot = math.sqrt(lam) * I_K
        A_aug = torch.cat([top, bot], dim=0)  # ((N_E+K), K)
        rhs = torch.cat(
            [
                torch.eye(N_E, dtype=Phi.dtype, device=Phi.device),
                torch.zeros((K, N_E), dtype=Phi.dtype, device=Phi.device),
            ],
            dim=0,
        )  # ((N_E+K), N_E)
        A = torch.linalg.lstsq(A_aug, rhs, rcond=None).solution  # (K, N_E)

    return A.to(torch.float32)


def gaussian_fit(e: Tensor, xanes: Tensor, widths_eV: List, stride: int) -> Tensor:
    basis = SpectralBasis(
        energies=e,
        widths_eV=widths_eV,
        normalize_atoms=True,
        stride=stride,
    )

    A = build_ridge_operator(basis.Phi, lam=1e-2)

    return xanes @ A.T
