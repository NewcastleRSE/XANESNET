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

"""Equivariant atom encoder with spherical-harmonics message passing for E3EE."""

from typing import cast

import torch
import torch.nn as nn
from e3nn import o3

from .basic import GaussianRBF, IrrepNorm
from .interactions import EquivariantInteractionBlock


class EquivariantAtomEncoder(nn.Module):
    """
    Equivariant atom encoder with spherical harmonics message passing.

    Args:
        max_z: Maximum atomic number (inclusive).
        cutoff: Message-passing cutoff radius in Angstrom.
        num_interactions: Number of equivariant interaction blocks.
        rbf_dim: Number of Gaussian RBF basis functions for edge distances.
        lmax: Maximum spherical-harmonics order.
        node_attr_dim: Embedding dimension for atomic-number embeddings.
        hidden_dim: Hidden dimension of radial weight MLPs.
        irreps_node: Output irreps of each interaction block,
            given as an e3nn irreps string (e.g. ``"32x0e+16x1o+8x2e"``).
        irreps_message: Intermediate irreps used for the tensor-product
            message before the gated non-linearity.
        residual_scale_init: Initial value of the learnable residual scale
            parameter in each interaction block.
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
        """Initialize ``EquivariantAtomEncoder``."""
        super().__init__()
        self.cutoff = cutoff
        self.rbf_dim = rbf_dim
        self.irreps_node = cast(o3.Irreps, o3.Irreps(irreps_node))
        self.irreps_message = cast(o3.Irreps, o3.Irreps(irreps_message))

        self.dist_rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.z_emb = nn.Embedding(max_z + 1, node_attr_dim)

        self.input_scalar_dim = node_attr_dim + 1
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
        mask: torch.Tensor,
        absorber_index: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Encode atoms into equivariant features via multi-layer message passing.

        Args:
            z: Atomic numbers (int64), shape ``(B, N)``.
            mask: Valid-atom mask, shape ``(B, N)``.
            absorber_index: Absorber atom index per sample (0..N-1), shape ``(B,)``.
            edge_src: Source flat indices into ``B*N``, shape ``(E,)``.
            edge_dst: Destination flat indices into ``B*N``, shape ``(E,)``.
            edge_weight: Edge lengths in **Angstrom** (PBC-correct), shape ``(E,)``.
            edge_vec: Edge displacement vectors in **Angstrom** (PBC-correct), shape ``(E, 3)``.

        Returns:
            Equivariant atom features of shape ``(B, N, irreps_dim)``.
        """
        device = z.device
        bsz, n_atoms = z.shape

        # Scalar input: element embedding + absorber flag (no distance).
        abs_flag = torch.zeros(bsz, n_atoms, dtype=torch.float32, device=device)
        batch_arange = torch.arange(bsz, device=device)
        abs_flag[batch_arange, absorber_index] = 1.0

        zf = self.z_emb(z)
        scalar_in = torch.cat([zf, abs_flag.unsqueeze(-1)], dim=-1)

        x = self.input_lin(scalar_in.reshape(bsz * n_atoms, self.input_scalar_dim))
        flat_mask = mask.reshape(bsz * n_atoms)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)

        # Edge features.
        if edge_src.numel() > 0:
            edge_len = edge_weight
            edge_dir = edge_vec / edge_len.unsqueeze(-1).clamp_min(1e-8)
            edge_rbf = self.dist_rbf(edge_len.clamp(max=self.cutoff))
            edge_sh = o3.spherical_harmonics(
                self.irreps_sh,
                edge_dir,
                normalize=True,
                normalization="component",
            )
        else:
            edge_len = torch.zeros(0, device=device, dtype=torch.float32)
            edge_rbf = torch.zeros(0, self.rbf_dim, device=device, dtype=torch.float32)
            edge_sh = torch.zeros(0, self.irreps_sh.dim, device=device, dtype=torch.float32)

        for block in self.blocks:
            x = block(
                x=x,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_sh=edge_sh,
                edge_rbf=edge_rbf,
                edge_len=edge_len,
            )
            # TODO apply mask after each block?

        x = self.out_norm(x)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)

        return x.view(bsz, n_atoms, self.irreps_node.dim)
