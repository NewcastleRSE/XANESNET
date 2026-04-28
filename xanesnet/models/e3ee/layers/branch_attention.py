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

"""Energy-conditioned invariant atom-attention branch for E3EE."""

import torch
import torch.nn as nn

from .basic import MLP, CosineCutoff, GaussianRBF


class EnergyConditionedAtomAttention(nn.Module):
    """
    Energy-conditioned attention over invariant atomwise features.

    Args:
        atom_dim: Dimension of the invariant per-atom features.
        e_dim: Dimension of the energy RBF embedding.
        hidden_dim: Hidden dimension of all internal MLPs.
        latent_dim: Output (latent) dimension; must be divisible by ``n_heads``.
        att_cutoff: Radius of the attention neighborhood graph in Angstrom.
        rbf_dim: Number of Gaussian RBF bases for the absorber->atom distance.
        max_z: Maximum atomic number supported by the element embedding.
        z_emb_dim: Embedding dimension for atomic numbers.
        n_heads: Number of attention heads.
    """

    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        hidden_dim: int,
        latent_dim: int,
        att_cutoff: float,
        rbf_dim: int = 16,
        max_z: int = 100,
        z_emb_dim: int = 32,
        n_heads: int = 4,
    ) -> None:
        """Initialize ``EnergyConditionedAtomAttention``."""
        super().__init__()
        if latent_dim % n_heads != 0:
            raise ValueError("latent_dim must be divisible by n_heads")

        self.atom_dim = atom_dim
        self.e_dim = e_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rbf_dim = rbf_dim
        self.att_cutoff = float(att_cutoff)
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads

        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.dist_rbf = GaussianRBF(0.0, self.att_cutoff, rbf_dim)
        self.value_envelope = CosineCutoff(self.att_cutoff)

        self.query_mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        # atom_static = [h, zr, is_abs, rbf(dist_from_absorber)].
        atom_static_dim = atom_dim + z_emb_dim + 1 + rbf_dim
        self.key_mlp = MLP(
            in_dim=atom_static_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )
        self.value_mlp = MLP(
            in_dim=atom_static_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        self.out_proj = MLP(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=2,
        )

        self.score_scale = self.head_dim**-0.5

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the last dimension into ``(n_heads, head_dim)``.

        Args:
            x: Tensor of shape ``(..., latent_dim)``.

        Returns:
            Tensor of shape ``(..., n_heads, head_dim)``.
        """
        new_shape = x.shape[:-1] + (self.n_heads, self.head_dim)
        return x.view(*new_shape)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        absorber_index: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy-conditioned attention over atoms and return latent.

        Args:
            h: Invariant atom features, shape ``(B, N, H)``.
            z: Atomic numbers, shape ``(B, N)``.
            mask: Valid-atom mask (encoder scope), shape ``(B, N)``.
            e_feat: Energy RBF features, shape ``(nE, dE)``.
            absorber_index: Absorber index per sample, shape ``(B,)``.
            att_dst: Flat destination indices into ``B*N`` (attention scope), shape ``(E_att,)``.
            att_dist: Absorber-to-atom distances in **Angstrom**, shape ``(E_att,)``.

        Returns:
            Latent tensor of shape ``(B, nE, latent_dim)``.
        """
        bsz, n_atoms, h_dim = h.shape
        n_energies, e_dim = e_feat.shape
        device = h.device
        flat = bsz * n_atoms

        batch_arange = torch.arange(bsz, device=device)
        h_abs = h[batch_arange, absorber_index, :]  # [B, H]

        q_in = torch.cat(
            [
                h_abs.unsqueeze(1).expand(bsz, n_energies, h_dim),
                e_feat.unsqueeze(0).expand(bsz, n_energies, e_dim),
            ],
            dim=-1,
        )
        q = self.query_mlp(q_in)  # [B, nE, L]

        # Per-atom distance + scope mask, derived from the att-graph.
        att_mask_flat = torch.zeros(flat, dtype=torch.bool, device=device)
        att_mask_flat[att_dst] = True
        att_dist_flat = torch.zeros(flat, dtype=h.dtype, device=device)
        att_dist_flat[att_dst] = att_dist.to(dtype=h.dtype)
        att_mask = att_mask_flat.view(bsz, n_atoms) & mask  # [B, N]
        rbf = self.dist_rbf(att_dist_flat.view(bsz, n_atoms))  # [B, N, rbf]

        zr = self.z_emb(z)  # [B, N, z_emb_dim]
        is_abs = torch.zeros(bsz, n_atoms, dtype=h.dtype, device=device)
        is_abs[batch_arange, absorber_index] = 1.0

        atom_static = torch.cat([h, zr, is_abs.unsqueeze(-1), rbf], dim=-1)
        k = self.key_mlp(atom_static)  # [B, N, L]
        v = self.value_mlp(atom_static)  # [B, N, L]
        env = self.value_envelope(att_dist_flat.view(bsz, n_atoms))  # [B, N]
        v = v * env.unsqueeze(-1)

        q = self._split_heads(q)  # [B, nE, nH, dH]
        k = self._split_heads(k)  # [B, N,  nH, dH]
        v = self._split_heads(v)  # [B, N,  nH, dH]

        scores = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1) * self.score_scale

        attn_mask = att_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        scores = scores.masked_fill(~attn_mask, -1e9)

        attn = torch.softmax(scores, dim=2)
        attn = attn * attn_mask.to(attn.dtype)
        # Renormalize after masking so weights sum to one over active atoms.
        attn = attn / attn.sum(dim=2, keepdim=True).clamp_min(1e-8)

        out = (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=2)
        out = out.reshape(bsz, n_energies, self.latent_dim)

        return self.out_proj(out)
