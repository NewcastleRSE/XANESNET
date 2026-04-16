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

from ..utils import build_absorber_relative_geometry
from .basic import MLP, CosineCutoff, GaussianRBF


class EnergyConditionedAtomAttention(nn.Module):
    """
    Faster energy-conditioned attention over atomwise invariant features.

    Queries are energy-dependent; keys/values are atom-dependent but
    energy-independent.
    """

    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        rbf_dim: int,
        hidden_dim: int,
        latent_dim: int,
        cutoff: float,
        max_z: int = 100,
        z_emb_dim: int = 32,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim % n_heads != 0:
            raise ValueError("latent_dim must be divisible by n_heads")

        self.atom_dim = atom_dim
        self.e_dim = e_dim
        self.rbf_dim = rbf_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cutoff = cutoff
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads

        self.rbf_fn = GaussianRBF(0.0, cutoff, rbf_dim)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)

        self.query_mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        atom_static_dim = atom_dim + z_emb_dim + rbf_dim + 3 + 1
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
        new_shape = x.shape[:-1] + (self.n_heads, self.head_dim)
        return x.view(*new_shape)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        absorber_index: int = 0,
        geom: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h:      [B, N, H] invariant atom features
            z:      [B, N] atomic numbers
            pos:    [B, N, 3] positions
            mask:   [B, N] valid-atom mask
            e_feat: [nE, dE] energy feature embedding
            absorber_index: absorber atom index
            geom:   precomputed absorber-relative geometry

        Returns:
            [B, nE, latent_dim]
        """
        bsz, n_atoms, h_dim = h.shape
        n_energies, e_dim = e_feat.shape

        if geom is None:
            geom = build_absorber_relative_geometry(pos=pos, mask=mask, absorber_index=absorber_index)

        r = geom["r"]  # [B, N]
        u = geom["u"]  # [B, N, 3]
        valid = mask & (r <= self.cutoff)

        h_abs = h[:, absorber_index, :]  # [B, H]
        q_in = torch.cat(
            [
                h_abs.unsqueeze(1).expand(bsz, n_energies, h_dim),
                e_feat.unsqueeze(0).expand(bsz, n_energies, e_dim),
            ],
            dim=-1,
        )
        q = self.query_mlp(q_in)  # [B, nE, L]

        zr = self.z_emb(z)  # [B, N, z_emb_dim]
        rr = self.rbf_fn(r.clamp(max=self.cutoff))  # [B, N, rbf_dim]
        is_abs = torch.zeros_like(r)
        is_abs[:, absorber_index] = 1.0

        atom_static = torch.cat([h, zr, rr, u, is_abs.unsqueeze(-1)], dim=-1)
        k = self.key_mlp(atom_static)  # [B, N, L]
        v = self.value_mlp(atom_static)  # [B, N, L]

        q = self._split_heads(q)  # [B, nE, nH, dH]
        k = self._split_heads(k)  # [B, N,  nH, dH]
        v = self._split_heads(v)  # [B, N,  nH, dH]

        scores = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1) * self.score_scale

        radial_bias = torch.log(self.cutoff_fn(r).clamp_min(1e-8)).unsqueeze(1).unsqueeze(-1)
        scores = scores + radial_bias

        attn_mask = valid.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~attn_mask, -1e9)

        attn = torch.softmax(scores, dim=2)
        attn = attn * attn_mask.to(attn.dtype)

        out = (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=2)
        out = out.reshape(bsz, n_energies, self.latent_dim)

        return self.out_proj(out)
