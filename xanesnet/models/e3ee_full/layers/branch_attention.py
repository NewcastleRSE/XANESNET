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

"""Energy-conditioned sparse atom-attention branch for E3EEFull."""

import torch
import torch.nn as nn

from .basic import MLP, CosineCutoff, GaussianRBF


def _scatter_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Numerically stable softmax over groups defined by ``index`` along dim 0.

    Args:
        scores: Floating-point scores of shape ``(E, ...)``.
        index: Long tensor of group ids in ``[0, dim_size)``, shape ``(E,)``.
        dim_size: Number of groups (typically ``B * N_max``).

    Returns:
        Softmax weights of shape ``(E, ...)`` normalized within each group.
    """
    if scores.numel() == 0:
        return scores

    extra = scores.shape[1:]
    flat_extra = int(torch.tensor(extra).prod().item()) if len(extra) > 0 else 1
    s = scores.reshape(scores.shape[0], flat_extra)

    src_max = torch.full((dim_size, flat_extra), float("-inf"), device=s.device, dtype=s.dtype)
    src_max.scatter_reduce_(
        0,
        index.view(-1, 1).expand_as(s),
        s,
        reduce="amax",
        include_self=True,
    )
    # Replace -inf (groups with no edges) with 0 to avoid NaN when subtracted.
    src_max = torch.where(torch.isinf(src_max), torch.zeros_like(src_max), src_max)

    s_shift = s - src_max[index]
    exp_s = torch.exp(s_shift)

    denom = torch.zeros(dim_size, flat_extra, device=s.device, dtype=s.dtype)
    denom.scatter_add_(0, index.view(-1, 1).expand_as(exp_s), exp_s)
    attn = exp_s / denom[index].clamp_min(1e-30)

    return attn.view(scores.shape[0], *extra)


class AllAtomAtomAttention(nn.Module):
    """Sparse energy-conditioned attention with one query per atom and one key/value per edge.

    The attention scope (which atoms each query may attend to) is supplied externally
    as the ``att_src``/``att_dst`` edge list -- typically a radius graph larger than the
    local encoder graph. RBF-encoded distances enter the keys/values, and values receive
    an additional cosine cutoff envelope. Softmax over each query's edge set provides renormalization.

    The optional ``absorber_mask`` lets the caller restrict queries to absorber atoms when
    ``use_absorber_mask`` is enabled in the parent model; other rows of the output are zeros.

    Args:
        atom_dim: Dimension of invariant per-atom features.
        e_dim: Dimension of the energy RBF embedding.
        hidden_dim: Hidden dimension of all internal MLPs.
        latent_dim: Output (latent) dimension; must be divisible by ``n_heads``.
        att_cutoff: Attention neighbourhood radius in **A**.
        rbf_dim: Number of Gaussian RBF bases for distance encoding.
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
        super().__init__()
        if latent_dim % n_heads != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by n_heads ({n_heads})")

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

        # Pair static features used for keys/values: dst atom features, dst
        # element embedding, RBF(distance) and a self-edge flag.
        pair_static_dim = atom_dim + z_emb_dim + 1 + rbf_dim
        self.key_mlp = MLP(
            in_dim=pair_static_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )
        self.value_mlp = MLP(
            in_dim=pair_static_dim,
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

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        att_src: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
        absorber_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute energy-conditioned attention over atoms and return latent.

        Args:
            h: Invariant atom features, shape ``(B, N, H)``.
            z: Atomic numbers, shape ``(B, N)``.
            mask: Valid-atom mask (encoder scope), shape ``(B, N)``.
            e_feat: Energy RBF features, shape ``(nE, dE)``.
            att_src: Flat source indices (queries) into ``B*N``, shape ``(E_att,)``.
            att_dst: Flat destination indices (keys/values) into ``B*N``, shape ``(E_att,)``.
            att_dist: Pair distances in **A**, shape ``(E_att,)``.
            absorber_mask: Optional ``(B, N)`` bool mask. When given, queries are
                restricted to atoms with ``True``; other rows are returned as zeros.

        Returns:
            Latent tensor of shape ``(B, N, nE, latent_dim)``.
        """
        bsz, n_atoms, h_dim = h.shape
        n_energies, e_dim = e_feat.shape
        device = h.device
        dtype = h.dtype
        flat = bsz * n_atoms

        h_flat = h.reshape(flat, h_dim)
        z_flat = z.reshape(flat)
        mask_flat = mask.reshape(flat)

        # Active queries.
        src_active = mask_flat.clone()
        if absorber_mask is not None:
            src_active = src_active & absorber_mask.reshape(flat)

        # Restrict to edges with active src and valid dst.
        # Normally the dataset and graph construction should ensure this already.
        edge_active = src_active[att_src] & mask_flat[att_dst]
        ea_src = att_src[edge_active]
        ea_dst = att_dst[edge_active]
        ea_dist = att_dist[edge_active].to(dtype=dtype)
        n_edges = ea_src.shape[0]

        out_flat = torch.zeros(flat, n_energies, self.latent_dim, device=device, dtype=dtype)

        if n_edges == 0:
            return self.out_proj(out_flat.view(bsz, n_atoms, n_energies, self.latent_dim))

        rbf = self.dist_rbf(ea_dist)  # [E, rbf]
        zr_d = self.z_emb(z_flat[ea_dst])  # [E, z_emb]
        is_self = (ea_src == ea_dst).to(dtype=dtype).unsqueeze(-1)  # [E, 1]
        h_dst = h_flat[ea_dst]  # [E, H]
        h_src = h_flat[ea_src]  # [E, H]

        pair_static = torch.cat([h_dst, zr_d, is_self, rbf], dim=-1)
        k_e = self.key_mlp(pair_static)  # [E, L]
        v_e = self.value_mlp(pair_static)  # [E, L]
        env_e = self.value_envelope(ea_dist).unsqueeze(-1)  # [E, 1]
        v_e = v_e * env_e

        # Per-edge query: depends on src and energy. Materialize [E, nE, L].
        q_in = torch.cat(
            [
                h_src.unsqueeze(1).expand(n_edges, n_energies, h_dim),
                e_feat.unsqueeze(0).expand(n_edges, n_energies, e_dim),
            ],
            dim=-1,
        )
        q_e = self.query_mlp(q_in)  # [E, nE, L]

        # Multi-head split.
        q_eh = q_e.view(n_edges, n_energies, self.n_heads, self.head_dim)
        k_eh = k_e.view(n_edges, self.n_heads, self.head_dim)
        v_eh = v_e.view(n_edges, self.n_heads, self.head_dim)

        # Scores per (edge, energy, head).
        scores = (q_eh * k_eh.unsqueeze(1)).sum(dim=-1) * self.score_scale  # [E, nE, nH]

        # Softmax over edges sharing the same src, per (energy, head).
        attn = _scatter_softmax(scores, ea_src, flat)  # [E, nE, nH]

        # Aggregate values per src.
        contrib = attn.unsqueeze(-1) * v_eh.unsqueeze(1)  # [E, nE, nH, dH]
        contrib_flat = contrib.reshape(n_edges, n_energies, self.latent_dim)

        out_flat.index_add_(0, ea_src, contrib_flat)

        out = out_flat.view(bsz, n_atoms, n_energies, self.latent_dim)
        return self.out_proj(out)
