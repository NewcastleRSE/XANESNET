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


class EnergyConditionedEquivariantAbsorberHead(nn.Module):
    """
    Late equivariant absorber head.

    Applies energy-conditioned irrep-wise modulation to the absorber
    equivariant feature, converts to invariants, then projects.
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

    def forward(self, h_abs_full: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_abs_full: [B, D] equivariant absorber features
            e_feat:     [nE, e_dim] energy features

        Returns:
            [B, nE, out_dim]
        """
        h_mod = self.mod(h_abs_full, e_feat)  # [B, nE, D]
        inv = invariant_features_from_irreps(h_mod, self.irreps_node)  # [B, nE, inv_dim]
        return self.out_mlp(inv)


class EnergyIrrepModulation(nn.Module):
    """
    Energy-conditioned scalar modulation of each irrep copy,
    preserving equivariance.
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
            x:      [B, D] equivariant features
            e_feat: [nE, e_dim] energy features

        Returns:
            [B, nE, D]
        """
        bsz, _ = x.shape

        gates = self.mlp(e_feat)  # [nE, n_copies]

        outs: list[torch.Tensor] = []
        xoff = 0
        goff = 0

        for mul, ir in self.irreps:
            dim = ir.dim
            block_dim = mul * dim

            xb = x[:, xoff : xoff + block_dim].view(bsz, mul, dim)
            gb = gates[:, goff : goff + mul]

            xb = xb.unsqueeze(1)  # [B, 1, mul, dim]
            gb = gb.unsqueeze(0).unsqueeze(-1)  # [1, nE, mul, 1]

            outs.append((xb * gb).reshape(bsz, e_feat.shape[0], block_dim))

            xoff += block_dim
            goff += mul

        return torch.cat(outs, dim=-1)
