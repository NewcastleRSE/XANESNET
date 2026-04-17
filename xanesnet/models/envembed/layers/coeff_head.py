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

from .encoder import init_mlp_weights


class ResidualPreLNBlock(nn.Module):
    """Pre-LayerNorm residual block: LN -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        init_mlp_weights(self.fc1)
        init_mlp_weights(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class CoeffHeadGroupedResidualPreLN(nn.Module):
    """
    Shared residual Pre-LN trunk over latent; per-width grouped linear heads.

    The trunk consists of ``depth`` ``ResidualPreLNBlock`` layers followed by a
    final LayerNorm.  Each group head is an independent ``nn.Linear`` mapping from
    the trunk output to ``k`` coefficients.  The outputs of all heads are
    concatenated to produce the final coefficient vector.
    """

    def __init__(
        self,
        latent_dim: int,
        k_groups: list[int],
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.k_groups = k_groups
        self.trunk = nn.Sequential(*[ResidualPreLNBlock(latent_dim, hidden, dropout) for _ in range(depth)])
        self.trunk_out_ln = nn.LayerNorm(latent_dim)
        self.group_heads = nn.ModuleList([nn.Linear(latent_dim, k) for k in self.k_groups])

        # Zero-init the group heads so initial predictions are near zero
        for head in self.group_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent representation from the encoder.

        Returns:
            (B, sum(k_groups)) concatenated coefficient predictions.
        """
        h = self.trunk(z)
        h = self.trunk_out_ln(h)
        outs = [head(h) for head in self.group_heads]
        return torch.cat(outs, dim=-1)
