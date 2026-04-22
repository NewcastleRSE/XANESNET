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

from typing import cast

import torch
import torch.nn as nn
from e3nn import o3

from ..utils import invariant_feature_dim, invariant_features_from_irreps
from .basic import MLP


class AllAtomEquivariantHead(nn.Module):
    """
    Late equivariant head applied to every atom.

    Applies energy-conditioned irrep-wise modulation to each atom's equivariant
    feature, converts to invariants, then projects.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.irreps_node = cast(o3.Irreps, o3.Irreps(irreps_node))
        self.mod = EnergyIrrepModulation(self.irreps_node, e_dim=e_dim, hidden_dim=hidden_dim)
        self.inv_dim = invariant_feature_dim(self.irreps_node)

        self.out_mlp = MLP(
            in_dim=self.inv_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, h_full: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_full: [B, N, D] equivariant features for every atom
            e_feat: [nE, e_dim]

        Returns:
            [B, N, nE, out_dim]
        """
        bsz, n_atoms, d = h_full.shape
        n_energies = e_feat.shape[0]

        h_flat = h_full.reshape(bsz * n_atoms, d)
        h_mod = self.mod(h_flat, e_feat)  # [B*N, nE, D]
        inv = invariant_features_from_irreps(h_mod, self.irreps_node)  # [B*N, nE, inv_dim]
        out = self.out_mlp(inv)  # [B*N, nE, out_dim]
        return out.view(bsz, n_atoms, n_energies, out.shape[-1])


class EnergyIrrepModulation(nn.Module):
    """
    Energy-conditioned scalar modulation of each irrep copy, preserving
    equivariance. Input ``x`` is a flat ``[M, D]`` tensor of equivariant
    features; the output broadcasts over energies into ``[M, nE, D]``.
    """

    def __init__(self, irreps: o3.Irreps, e_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.irreps = cast(o3.Irreps, o3.Irreps(irreps))
        self.n_copies = sum(mul for mul, _ in self.irreps)

        self.mlp = MLP(
            in_dim=e_dim,
            hidden_dim=hidden_dim,
            out_dim=self.n_copies,
            n_layers=3,
        )

    def forward(self, x: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      [M, D]  equivariant features (M = B * N for our use case)
            e_feat: [nE, e_dim]

        Returns:
            [M, nE, D]
        """
        m = x.shape[0]
        n_energies = e_feat.shape[0]

        gates = self.mlp(e_feat)  # [nE, n_copies]

        outs: list[torch.Tensor] = []
        xoff = 0
        goff = 0

        for mul, ir in self.irreps:
            dim = ir.dim
            block_dim = mul * dim

            xb = x[:, xoff : xoff + block_dim].view(m, mul, dim)
            gb = gates[:, goff : goff + mul]

            xb = xb.unsqueeze(1)  # [M, 1, mul, dim]
            gb = gb.unsqueeze(0).unsqueeze(-1)  # [1, nE, mul, 1]

            outs.append((xb * gb).reshape(m, n_energies, block_dim))

            xoff += block_dim
            goff += mul

        return torch.cat(outs, dim=-1)
