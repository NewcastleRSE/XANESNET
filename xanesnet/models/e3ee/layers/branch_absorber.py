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

"""Invariant absorber branch for E3EE."""

import torch
import torch.nn as nn

from .basic import MLP


class EnergyConditionedAbsorberBranch(nn.Module):
    """Energy-dependent absorber branch based on invariant absorber features.

    Args:
        atom_dim: Dimension of the invariant absorber feature vector.
        e_dim: Dimension of the energy RBF embedding.
        hidden_dim: Hidden dimension of the MLP.
        out_dim: Output (latent) dimension.
    """

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

    def forward(self, h_abs: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """Compute absorber branch latent from absorber features and energy embedding.

        Args:
            h_abs: Absorber invariant features, shape ``(B, H)``.
            e_feat: Energy RBF features, shape ``(nE, dE)``.

        Returns:
            Latent tensor of shape ``(B, nE, latent_dim)``.
        """
        bsz, h_dim = h_abs.shape
        n_energies, e_dim = e_feat.shape

        ha = h_abs.unsqueeze(1).expand(bsz, n_energies, h_dim)
        ef = e_feat.unsqueeze(0).expand(bsz, n_energies, e_dim)
        return self.mlp(torch.cat([ha, ef], dim=-1))
