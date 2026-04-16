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

from ..utils import build_absorber_relative_geometry
from .basic import GaussianRBF, IrrepNorm
from .interactions import EquivariantInteractionBlock
from .radiusgraph import BatchedRadiusGraphBuilder


class EquivariantAtomEncoder(nn.Module):
    """
    Equivariant atom encoder with spherical harmonics message passing.

    Produces per-atom features in the given irreps representation.
    """

    def __init__(
        self,
        max_z: int,
        cutoff: float,
        num_interactions: int,
        rbf_dim: int,
        lmax: int,
        node_attr_dim: int,
        hidden_dim: int,
        irreps_node: str,
        irreps_message: str,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.rbf_dim = rbf_dim
        self.irreps_node = cast(o3.Irreps, o3.Irreps(irreps_node))
        self.irreps_message = cast(o3.Irreps, o3.Irreps(irreps_message))

        self.graph_builder = BatchedRadiusGraphBuilder(cutoff=cutoff)
        self.dist_rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.z_emb = nn.Embedding(max_z + 1, node_attr_dim)

        self.input_scalar_dim = node_attr_dim + 1 + rbf_dim
        self.input_lin = o3.Linear(
            irreps_in=o3.Irreps(f"{self.input_scalar_dim}x0e"),
            irreps_out=self.irreps_node,
        )

        self.irreps_sh = cast(o3.Irreps, o3.Irreps.spherical_harmonics(lmax))

        self.blocks = nn.ModuleList(
            [
                EquivariantInteractionBlock(
                    irreps_node=str(self.irreps_node),
                    irreps_sh=str(self.irreps_sh),
                    irreps_message=str(self.irreps_message),
                    rbf_dim=rbf_dim,
                    radial_hidden_dim=hidden_dim,
                    cutoff=cutoff,
                    residual_scale_init=residual_scale_init,
                )
                for _ in range(num_interactions)
            ]
        )

        self.out_norm = IrrepNorm(self.irreps_node)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        absorber_index: int = 0,
        geom: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:    [B, N] atomic numbers
            pos:  [B, N, 3] positions
            mask: [B, N] valid-atom mask
            absorber_index: index of the absorber atom
            geom: precomputed absorber-relative geometry (optional)

        Returns:
            [B, N, irreps_dim] equivariant atom features
        """
        device = pos.device
        bsz, n_atoms = z.shape

        if geom is None:
            geom = build_absorber_relative_geometry(pos=pos, mask=mask, absorber_index=absorber_index)

        r_abs = geom["r"]

        abs_flag = torch.zeros_like(z, dtype=pos.dtype)
        abs_flag[:, absorber_index] = 1.0

        zf = self.z_emb(z)
        rf = self.dist_rbf(r_abs.clamp(max=self.cutoff))

        scalar_in = torch.cat([zf, abs_flag.unsqueeze(-1), rf], dim=-1)
        x = self.input_lin(scalar_in.reshape(bsz * n_atoms, self.input_scalar_dim))
        flat_mask = mask.reshape(bsz * n_atoms)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)

        edge_src, edge_dst, edge_vec = self.graph_builder(pos, mask)

        if edge_vec.numel() > 0:
            edge_len = torch.linalg.norm(edge_vec, dim=-1)
            edge_dir = edge_vec / edge_len.unsqueeze(-1).clamp_min(1e-8)
            edge_rbf = self.dist_rbf(edge_len.clamp(max=self.cutoff))
            edge_sh = o3.spherical_harmonics(
                self.irreps_sh,
                edge_dir,
                normalize=True,
                normalization="component",
            )
        else:
            edge_len = torch.zeros(0, device=device, dtype=pos.dtype)
            edge_rbf = torch.zeros(0, self.rbf_dim, device=device, dtype=pos.dtype)
            edge_sh = torch.zeros(0, self.irreps_sh.dim, device=device, dtype=pos.dtype)

        for block in self.blocks:
            x = block(
                x=x,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_sh=edge_sh,
                edge_rbf=edge_rbf,
                edge_len=edge_len,
            )

        x = self.out_norm(x)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)

        return x.view(bsz, n_atoms, self.irreps_node.dim)
