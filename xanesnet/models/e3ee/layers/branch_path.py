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

"""3-body path aggregation branch for E3EE."""

import torch
import torch.nn as nn

from .basic import MLP, CosineCutoff, GaussianRBF


class PairElementEnergyScattering(nn.Module):
    """Energy-conditioned element-pair scattering features.

    Operates on flat per-path tensors ``(P, ...)`` rather than padded ``(B, n_paths, ...)``.

    Args:
        max_z: Maximum atomic number supported by the element embedding.
        z_emb_dim: Embedding dimension for atomic numbers.
        e_dim: Dimension of the energy RBF embedding.
        hidden_dim: Hidden dimension of the output MLP.
        out_dim: Output feature dimension per path.
    """

    def __init__(
        self,
        max_z: int,
        z_emb_dim: int,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        """Initialize ``PairElementEnergyScattering``."""
        super().__init__()
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.mlp = MLP(
            in_dim=2 * z_emb_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(
        self,
        z_j: torch.Tensor,
        z_k: torch.Tensor,
        e_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-path element-pair energy features.

        Args:
            z_j: Atomic numbers of the j leg, shape ``(P,)``.
            z_k: Atomic numbers of the k leg, shape ``(P,)``.
            e_feat: Energy RBF features, shape ``(nE, e_dim)``.

        Returns:
            Per-path energy features of shape ``(P, nE, out_dim)``.
        """
        n_paths = z_j.shape[0]
        n_energies, e_dim = e_feat.shape

        ej = self.z_emb(z_j).unsqueeze(1).expand(n_paths, n_energies, -1)
        ek = self.z_emb(z_k).unsqueeze(1).expand(n_paths, n_energies, -1)
        ef = e_feat.unsqueeze(0).expand(n_paths, n_energies, e_dim)

        return self.mlp(torch.cat([ej, ek, ef], dim=-1))


class AbsorberPathAggregator(nn.Module):
    """
    3-body absorber-centred path aggregator for paths (absorber, j, k).

    Consumes precomputed flat triplet scalars (``r0j``, ``r0k``, ``rjk``,
    ``cos(angle)``) and per-path flat atom indices into the padded ``B * N``
    node layout, together with ``path_batch`` for scatter-aggregating into the
    batch dimension.

    Args:
        atom_dim: Dimension of invariant per-atom features.
        rbf_dim: Number of Gaussian RBF bases for distance encoding.
        geom_hidden_dim: Hidden dimension of the geometry MLP.
        scatter_dim: Intermediate scatter feature dimension.
        out_dim: Output (latent) dimension.
        cutoff: Radial cutoff used for the cosine envelope weights.
    """

    def __init__(
        self,
        atom_dim: int,
        rbf_dim: int,
        geom_hidden_dim: int,
        scatter_dim: int,
        out_dim: int,
        cutoff: float,
    ) -> None:
        """Initialize ``AbsorberPathAggregator``."""
        super().__init__()
        self.cutoff = cutoff
        self.out_dim = out_dim
        self.rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.cutoff_fn = CosineCutoff(cutoff)

        # 2 * atom_dim (hj, hk) + 3 * rbf_dim (r0j, r0k, rjk) + 1 (cos angle)
        self.geom_mlp = MLP(
            in_dim=2 * atom_dim + 3 * rbf_dim + 1,
            hidden_dim=geom_hidden_dim,
            out_dim=scatter_dim,
            n_layers=3,
        )

        self.out_proj = MLP(
            in_dim=scatter_dim,
            hidden_dim=geom_hidden_dim,
            out_dim=out_dim,
            n_layers=2,
        )

    def forward(
        self,
        h_flat: torch.Tensor,
        z_flat: torch.Tensor,
        pair_elem_energy: PairElementEnergyScattering,
        e_feat: torch.Tensor,
        path_j: torch.Tensor,
        path_k: torch.Tensor,
        path_r0j: torch.Tensor,
        path_r0k: torch.Tensor,
        path_rjk: torch.Tensor,
        path_cosangle: torch.Tensor,
        path_batch: torch.Tensor,
        bsz: int,
    ) -> torch.Tensor:
        """Aggregate path contributions into per-(sample, energy) latent vectors.

        Args:
            h_flat: Invariant atom features (flattened), shape ``(B*N, atom_dim)``.
            z_flat: Atomic numbers (flattened), shape ``(B*N,)``.
            pair_elem_energy: Pre-built element-pair energy module.
            e_feat: Energy RBF features, shape ``(nE, e_dim)``.
            path_j: Flat atom indices for leg j (into B*N), shape ``(P,)``.
            path_k: Flat atom indices for leg k (into B*N), shape ``(P,)``.
            path_r0j: Absorber-to-j distance in **Angstrom**, shape ``(P,)``.
            path_r0k: Absorber-to-k distance in **Angstrom**, shape ``(P,)``.
            path_rjk: j-to-k distance in **Angstrom**, shape ``(P,)``.
            path_cosangle: Cosine of the j-absorber-k angle, shape ``(P,)``.
            path_batch: Batch index per path (0..B-1), shape ``(P,)``.
            bsz: Batch size.

        Returns:
            Latent tensor of shape ``(B, nE, out_dim)``.
        """
        device = h_flat.device
        n_energies = e_feat.shape[0]
        n_paths = path_j.shape[0]

        if n_paths == 0:
            return torch.zeros(bsz, n_energies, self.out_dim, device=device, dtype=h_flat.dtype)

        hj = h_flat[path_j]
        hk = h_flat[path_k]

        f0j = self.rbf(path_r0j.clamp(max=self.cutoff))
        f0k = self.rbf(path_r0k.clamp(max=self.cutoff))
        fjk = self.rbf(path_rjk.clamp(max=self.cutoff))

        geom_in = torch.cat([hj, hk, f0j, f0k, fjk, path_cosangle.unsqueeze(-1)], dim=-1)
        g_geom = self.geom_mlp(geom_in)  # [P, scatter_dim]

        zj = z_flat[path_j]
        zk = z_flat[path_k]
        g_elem = pair_elem_energy(zj, zk, e_feat)  # [P, nE, scatter_dim]

        cutoff_w = (
            (self.cutoff_fn(path_r0j) * self.cutoff_fn(path_r0k) * self.cutoff_fn(path_rjk)).unsqueeze(-1).unsqueeze(-1)
        )  # [P, 1, 1]

        contrib = g_geom.unsqueeze(1) * g_elem  # [P, nE, scatter_dim]
        contrib = contrib * cutoff_w

        scatter_dim = contrib.shape[-1]
        agg = torch.zeros(bsz, n_energies, scatter_dim, device=device, dtype=contrib.dtype)
        norm = torch.zeros(bsz, 1, 1, device=device, dtype=contrib.dtype)

        batch_expand = path_batch.view(-1, 1, 1).expand(-1, n_energies, scatter_dim)
        agg.scatter_add_(0, batch_expand, contrib)

        norm_expand = path_batch.view(-1, 1, 1)
        norm.scatter_add_(0, norm_expand, cutoff_w.squeeze(-1).unsqueeze(-1))

        agg = agg / norm.clamp_min(1e-8)

        return self.out_proj(agg)
