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
import torch.nn.functional as F

from .basic import MLP


class AllAtomAtomAttention(nn.Module):
    """
    Energy-conditioned global attention over invariant atomwise features with
    one query per (atom, energy) pair.

    Queries depend on the querying atom's own invariant features and the
    energy; keys and values depend only on the atom (element embedding +
    invariant features) and are shared across all query atoms. Uses
    ``F.scaled_dot_product_attention`` so CUDA FlashAttention can be used when
    available.
    """

    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_z: int = 100,
        z_emb_dim: int = 32,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim % n_heads != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by n_heads ({n_heads})")

        self.atom_dim = atom_dim
        self.e_dim = e_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads

        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)

        self.query_mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        # atom_static = [h, zr] — no absorber flag.
        atom_static_dim = atom_dim + z_emb_dim
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

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h:      [B, N, H] invariant atom features
            z:      [B, N]    atomic numbers
            mask:   [B, N]    valid-atom mask (attention scope for keys)
            e_feat: [nE, dE]  energy feature embedding

        Returns:
            [B, N, nE, latent_dim]
        """
        bsz, n_atoms, h_dim = h.shape
        n_energies, e_dim = e_feat.shape

        # Query input: [B, N, nE, H+dE]
        h_q = h.unsqueeze(2).expand(bsz, n_atoms, n_energies, h_dim)
        e_q = e_feat.view(1, 1, n_energies, e_dim).expand(bsz, n_atoms, n_energies, e_dim)
        q = self.query_mlp(torch.cat([h_q, e_q], dim=-1))  # [B, N, nE, L]

        zr = self.z_emb(z)  # [B, N, z_emb_dim]
        atom_static = torch.cat([h, zr], dim=-1)  # [B, N, H+z_emb_dim]
        k = self.key_mlp(atom_static)  # [B, N, L]
        v = self.value_mlp(atom_static)  # [B, N, L]

        # Reshape for SDPA: [B, nH, Lq, dH] / [B, nH, Lk, dH]
        q = q.reshape(bsz, n_atoms * n_energies, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, n_atoms, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, n_atoms, self.n_heads, self.head_dim).transpose(1, 2)

        # Key mask: broadcast [B, 1, 1, N]. True = keep.
        attn_mask = mask.view(bsz, 1, 1, n_atoms)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # [B, nH, N*nE, dH] -> [B, N, nE, L]
        out = out.transpose(1, 2).contiguous().view(bsz, n_atoms, n_energies, self.latent_dim)

        return self.out_proj(out)
