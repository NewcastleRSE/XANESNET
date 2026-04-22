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
import torch.nn as nn

from .basic import MLP


class AllAtomEnergyBranch(nn.Module):
    """Energy-dependent branch applied to every atom's invariant features."""

    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, h_all: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_all:  [B, N, H] invariant features for every atom
            e_feat: [nE, dE] energy features

        Returns:
            [B, N, nE, out_dim]
        """
        bsz, n_atoms, h_dim = h_all.shape
        n_energies, e_dim = e_feat.shape

        ha = h_all.unsqueeze(2).expand(bsz, n_atoms, n_energies, h_dim)
        ef = e_feat.view(1, 1, n_energies, e_dim).expand(bsz, n_atoms, n_energies, e_dim)
        return self.mlp(torch.cat([ha, ef], dim=-1))
