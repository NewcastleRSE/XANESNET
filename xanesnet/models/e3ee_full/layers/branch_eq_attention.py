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

"""Energy-conditioned equivariant atom-attention branch for E3EEFull."""

from typing import cast

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct

from ..utils import invariant_feature_dim, invariant_features_from_irreps
from .basic import MLP, CosineCutoff, GaussianRBF, RadialMLP
from .branch_attention import _scatter_softmax
from .branch_equivariant import EnergyIrrepModulation


class AllAtomEquivariantAtomAttention(nn.Module):
    """All-atom equivariant counterpart of :class:`AllAtomAtomAttention`.

    Each atom (or just the absorber atoms when ``use_absorber_mask`` is set)
    queries its attention-graph neighbours. The value carried along each edge is
    an E(3)-equivariant feature built from spherical harmonics of the
    src->dst unit vector mixed with the dst atom's full equivariant features via
    a FullyConnectedTensorProduct, then modulated per-energy with
    :class:`EnergyIrrepModulation`. Aggregated equivariant features are
    converted to invariants and projected to ``latent_dim``.

    Args:
        atom_dim: Dimension of invariant per-atom features (used for scoring).
        irreps_node: Irreps of the full equivariant atom features (TP input).
        e_dim: Dimension of the energy RBF embedding.
        hidden_dim: Hidden dimension of all internal MLPs.
        latent_dim: Output (latent) dimension; must be divisible by ``n_heads``.
        att_cutoff: Attention neighbourhood radius in **A**.
        attention_lmax: Maximum spherical-harmonics order for bond directions.
        attention_irreps: Target irreps of the equivariant values (e.g. ``"32x0e+16x1o"``).
        rbf_dim: Number of Gaussian RBF bases for distance encoding.
        max_z: Maximum atomic number supported by the element embedding.
        z_emb_dim: Embedding dimension for atomic numbers.
        n_heads: Number of attention heads used for invariant scoring.
    """

    def __init__(
        self,
        atom_dim: int,
        irreps_node: o3.Irreps,
        e_dim: int,
        hidden_dim: int,
        latent_dim: int,
        att_cutoff: float,
        attention_lmax: int,
        attention_irreps: str,
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

        self.sh_irreps = cast(o3.Irreps, o3.Irreps.spherical_harmonics(int(attention_lmax)))
        self.out_irreps = cast(o3.Irreps, o3.Irreps(attention_irreps))
        self.irreps_node = cast(o3.Irreps, o3.Irreps(irreps_node))

        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.dist_rbf = GaussianRBF(0.0, self.att_cutoff, rbf_dim)

        self.value_envelope = CosineCutoff(self.att_cutoff)

        # Equivariant value: TP(full encoder irreps at dst, SH(u_src->dst)) with
        # weights conditioned on (z_dst, is_self, RBF(dist)).
        self.value_tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.sh_irreps,
            self.out_irreps,
            shared_weights=False,
        )
        weight_in_dim = z_emb_dim + 1 + rbf_dim
        self.value_weight_mlp = RadialMLP(weight_in_dim, hidden_dim, self.value_tp.weight_numel)

        # Energy modulation (energy-independent equivariant value \u2192 [E, nE, irrep_dim]).
        self.energy_mod = EnergyIrrepModulation(self.out_irreps, e_dim=e_dim, hidden_dim=hidden_dim)

        # Invariant scoring (per-edge q\u00b7k, multi-head averaged).
        self.query_mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )
        pair_static_dim = atom_dim + z_emb_dim + 1 + rbf_dim
        self.key_mlp = MLP(
            in_dim=pair_static_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        self.inv_dim = invariant_feature_dim(self.out_irreps)
        self.out_mlp = MLP(
            in_dim=self.inv_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        self.score_scale = self.head_dim**-0.5

    def forward(
        self,
        h: torch.Tensor,
        h_full: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        att_src: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
        att_vec: torch.Tensor,
        absorber_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute equivariant energy-conditioned attention over atoms and return latent.

        Args:
            h: Invariant atom features (used for scoring), shape ``(B, N, H)``.
            h_full: Full equivariant atom features (TP input for values), shape ``(B, N, irreps_node.dim)``.
            z: Atomic numbers, shape ``(B, N)``.
            mask: Valid-atom mask, shape ``(B, N)``.
            e_feat: Energy RBF features, shape ``(nE, dE)``.
            att_src: Flat source indices (queries) into ``B*N``, shape ``(E_att,)``.
            att_dst: Flat destination indices (keys/values) into ``B*N``, shape ``(E_att,)``.
            att_dist: Pair distances in **A**, shape ``(E_att,)``.
            att_vec: Pair displacement vectors (src->dst) in **A**, shape ``(E_att, 3)``.
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
        h_full_flat = h_full.reshape(flat, self.irreps_node.dim)
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
        ea_vec = att_vec[edge_active].to(dtype=dtype)
        n_edges = ea_src.shape[0]

        out_inv = torch.zeros(flat, n_energies, self.out_irreps.dim, device=device, dtype=dtype)

        if n_edges == 0:
            zero_inv = invariant_features_from_irreps(out_inv, self.out_irreps)
            return self.out_mlp(zero_inv).view(bsz, n_atoms, n_energies, self.latent_dim)

        # Unit vector (zero-vector self-edges -> SH(0) is well-defined as constant 0e=1, 1o=0).
        eps_dist = ea_dist.clamp_min(1e-8)
        u = ea_vec / eps_dist.unsqueeze(-1)
        # For self-edges we want SH = constant l=0 only. spherical_harmonics on
        # the zero direction with normalize=True is undefined; gate by is_self.
        is_self = (ea_src == ea_dst).to(dtype=dtype).unsqueeze(-1)  # [E, 1]
        u_safe = torch.where(is_self.bool().expand_as(u), torch.zeros_like(u), u)
        # For the masked (self) edges we replace u with z-axis direction (any
        # fixed direction). The SH features for self-edges are then masked out
        # to keep only the l=0 component below.
        # Simpler: feed unit z for self-edges, zero out higher-l SH components.
        u_for_sh = torch.where(
            is_self.bool().expand_as(u),
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(u),
            u_safe,
        )
        sh = o3.spherical_harmonics(self.sh_irreps, u_for_sh, normalize=True, normalization="component")
        # Zero out l>0 SH components for self-edges.
        sh = self._mask_l0_only(sh, is_self.squeeze(-1).bool())

        rbf = self.dist_rbf(ea_dist)  # [E, rbf]
        zr_d = self.z_emb(z_flat[ea_dst])  # [E, z_emb]
        h_dst = h_flat[ea_dst]  # [E, H]
        h_src = h_flat[ea_src]  # [E, H]
        h_full_dst = h_full_flat[ea_dst]  # [E, irreps_node.dim]

        # Equivariant value (energy-independent).
        weight_in = torch.cat([zr_d, is_self, rbf], dim=-1)
        tp_weights = self.value_weight_mlp(weight_in)  # [E, weight_numel]
        v_irrep = self.value_tp(h_full_dst, sh, tp_weights)  # [E, out_irreps.dim]
        env_e = self.value_envelope(ea_dist).unsqueeze(-1)  # [E, 1]
        v_irrep = v_irrep * env_e

        # Per-energy modulation -> [E, nE, out_irreps.dim]
        v_mod = self.energy_mod(v_irrep, e_feat)

        # Invariant scoring per (edge, energy, head) \u2192 mean over heads.
        pair_static = torch.cat([h_dst, zr_d, is_self, rbf], dim=-1)
        k_e = self.key_mlp(pair_static)
        q_in = torch.cat(
            [
                h_src.unsqueeze(1).expand(n_edges, n_energies, h_dim),
                e_feat.unsqueeze(0).expand(n_edges, n_energies, e_dim),
            ],
            dim=-1,
        )
        q_e = self.query_mlp(q_in)  # [E, nE, L]
        q_eh = q_e.view(n_edges, n_energies, self.n_heads, self.head_dim)
        k_eh = k_e.view(n_edges, self.n_heads, self.head_dim)
        scores = (q_eh * k_eh.unsqueeze(1)).sum(dim=-1) * self.score_scale  # [E, nE, nH]
        scores = scores.mean(dim=-1)  # [E, nE]

        attn = _scatter_softmax(scores, ea_src, flat)  # [E, nE]

        # Aggregate equivariant value contributions per src.
        contrib = attn.unsqueeze(-1) * v_mod  # [E, nE, irrep_dim]
        out_inv.index_add_(0, ea_src, contrib)

        out_irrep = out_inv.view(bsz * n_atoms, n_energies, self.out_irreps.dim)
        inv = invariant_features_from_irreps(out_irrep, self.out_irreps)  # [B*N, nE, inv_dim]
        out = self.out_mlp(inv).view(bsz, n_atoms, n_energies, self.latent_dim)
        return out

    def _mask_l0_only(self, sh: torch.Tensor, mask_self: torch.Tensor) -> torch.Tensor:
        """Zero out l>0 components of ``sh`` for rows in ``mask_self``."""
        if not mask_self.any():
            return sh
        out = sh.clone()
        offset = 0
        for _mul, ir in self.sh_irreps:
            dim = ir.dim
            if ir.l > 0:
                out[mask_self, offset : offset + dim] = 0.0
            offset += dim
        return out
