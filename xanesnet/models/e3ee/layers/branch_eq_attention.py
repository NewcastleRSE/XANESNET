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

"""Energy-conditioned equivariant atom-attention branch for E3EE."""

from typing import cast

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct

from ..utils import invariant_feature_dim, invariant_features_from_irreps
from .basic import MLP, CosineCutoff, GaussianRBF, RadialMLP
from .branch_equivariant import EnergyIrrepModulation


class EnergyConditionedEquivariantAtomAttention(nn.Module):
    """
    Equivariant counterpart of :class:`EnergyConditionedAtomAttention`.

    Args:
        atom_dim: Dimension of invariant per-atom features (used for scoring).
        irreps_node: Irreps of the full equivariant atom features (TP input).
        e_dim: Dimension of the energy RBF embedding.
        hidden_dim: Hidden dimension of all internal MLPs.
        latent_dim: Output (latent) dimension; must be divisible by ``n_heads``.
        att_cutoff: Radius of the attention neighborhood graph in Angstrom.
        attention_lmax: Maximum spherical-harmonics order for bond directions.
        attention_irreps: Target irreps of the equivariant values (e.g. ``"32x0e+16x1o"``).
        rbf_dim: Number of Gaussian RBF bases for the absorber->atom distance.
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
        """Initialize ``EnergyConditionedEquivariantAtomAttention``."""
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

        self.sh_irreps = cast(o3.Irreps, o3.Irreps.spherical_harmonics(int(attention_lmax)))
        self.out_irreps = cast(o3.Irreps, o3.Irreps(attention_irreps))
        self.irreps_node = cast(o3.Irreps, o3.Irreps(irreps_node))

        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.dist_rbf = GaussianRBF(0.0, self.att_cutoff, rbf_dim)
        self.value_envelope = CosineCutoff(self.att_cutoff)

        # Equivariant value: TP(full encoder irreps, SH(u)) -> out irreps,
        # with weights conditioned on the per-atom RBF / element / absorber flag.
        self.value_tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.sh_irreps,
            self.out_irreps,
            shared_weights=False,
        )
        weight_in_dim = z_emb_dim + 1 + rbf_dim
        self.value_weight_mlp = RadialMLP(weight_in_dim, hidden_dim, self.value_tp.weight_numel)

        # Energy modulation of the (energy-independent) equivariant value.
        self.energy_mod = EnergyIrrepModulation(self.out_irreps, e_dim=e_dim, hidden_dim=hidden_dim)

        # Invariant scoring, same recipe as the invariant attention.
        self.query_mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )
        atom_static_dim = atom_dim + z_emb_dim + 1 + rbf_dim
        self.key_mlp = MLP(
            in_dim=atom_static_dim,
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
        h_full: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        absorber_index: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
        att_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Compute equivariant energy-conditioned attention over atoms and return latent.

        Args:
            h: Invariant atom features (used for scoring), shape ``(B, N, H)``.
            h_full: Full equivariant atom features (TP input for values),
                shape ``(B, N, irreps_node.dim)``.
            z: Atomic numbers, shape ``(B, N)``.
            mask: Valid-atom mask, shape ``(B, N)``.
            e_feat: Energy RBF features, shape ``(nE, dE)``.
            absorber_index: Absorber index per sample, shape ``(B,)``.
            att_dst: Flat destination indices into ``B*N``, shape ``(E_att,)``.
            att_dist: Absorber-to-atom distances in **Angstrom**, shape ``(E_att,)``.
            att_vec: Absorber-to-atom displacement vectors in **Angstrom**, shape ``(E_att, 3)``.

        Returns:
            Latent tensor of shape ``(B, nE, latent_dim)``.
        """
        bsz, n_atoms, h_dim = h.shape
        n_energies, e_dim = e_feat.shape
        device = h.device
        flat = bsz * n_atoms

        # Per-atom attention scope, distance and unit vector.
        att_mask_flat = torch.zeros(flat, dtype=torch.bool, device=device)
        att_mask_flat[att_dst] = True
        att_dist_flat = torch.zeros(flat, dtype=h.dtype, device=device)
        att_dist_flat[att_dst] = att_dist.to(dtype=h.dtype)
        att_vec_flat = torch.zeros(flat, 3, dtype=h.dtype, device=device)
        att_vec_flat[att_dst] = att_vec.to(dtype=h.dtype)
        att_mask = att_mask_flat.view(bsz, n_atoms) & mask  # [B, N]

        eps_dist = att_dist_flat.clamp_min(1e-8)
        u = att_vec_flat / eps_dist.unsqueeze(-1)  # [flat, 3]

        sh = o3.spherical_harmonics(self.sh_irreps, u, normalize=True, normalization="component")
        rbf_flat = self.dist_rbf(att_dist_flat)  # [flat, rbf_dim]

        # Element / absorber flag features.
        zr = self.z_emb(z)  # [B, N, z_emb_dim]
        zr_flat = zr.view(flat, -1)
        batch_arange = torch.arange(bsz, device=device)
        is_abs = torch.zeros(bsz, n_atoms, dtype=h.dtype, device=device)
        is_abs[batch_arange, absorber_index] = 1.0
        is_abs_flat = is_abs.view(flat, 1)

        # Equivariant value (energy-independent).
        weight_in = torch.cat([zr_flat, is_abs_flat, rbf_flat], dim=-1)
        tp_weights = self.value_weight_mlp(weight_in)  # [flat, weight_numel]
        h_full_flat = h_full.reshape(flat, self.irreps_node.dim)
        v_irrep = self.value_tp(h_full_flat, sh, tp_weights)  # [flat, out_irreps.dim]
        # Zero out value at atoms outside the attention scope so they cannot
        # leak through the (renormalized) softmax (which still has them).
        v_irrep = v_irrep * att_mask_flat.unsqueeze(-1).to(v_irrep.dtype)
        env = self.value_envelope(att_dist_flat).unsqueeze(-1)  # [flat, 1]
        v_irrep = v_irrep * env

        # Per-energy modulation -> [flat, nE, out_irreps.dim]
        v_mod = self.energy_mod(v_irrep, e_feat)
        v_mod = v_mod.view(bsz, n_atoms, n_energies, self.out_irreps.dim)

        # Invariant scoring.
        h_abs = h[batch_arange, absorber_index, :]  # [B, H]
        q_in = torch.cat(
            [
                h_abs.unsqueeze(1).expand(bsz, n_energies, h_dim),
                e_feat.unsqueeze(0).expand(bsz, n_energies, e_dim),
            ],
            dim=-1,
        )
        q = self.query_mlp(q_in)  # [B, nE, L]

        atom_static = torch.cat(
            [h, zr, is_abs.unsqueeze(-1), rbf_flat.view(bsz, n_atoms, -1)],
            dim=-1,
        )
        k = self.key_mlp(atom_static)  # [B, N, L]

        q_h = self._split_heads(q)  # [B, nE, nH, dH]
        k_h = self._split_heads(k)  # [B, N,  nH, dH]
        scores = (q_h.unsqueeze(2) * k_h.unsqueeze(1)).sum(dim=-1) * self.score_scale  # [B, nE, N, nH]
        # Average heads to a single attention weight per (B, nE, N).
        scores = scores.mean(dim=-1)  # [B, nE, N]

        attn_mask = att_mask.unsqueeze(1)  # [B, 1, N]
        scores = scores.masked_fill(~attn_mask, -1e9)
        attn = torch.softmax(scores, dim=2)
        attn = attn * attn_mask.to(attn.dtype)
        attn = attn / attn.sum(dim=2, keepdim=True).clamp_min(1e-8)  # [B, nE, N]

        # Aggregate equivariant value: sum_n attn[b, e, n] * v_mod[b, n, e, :]
        # -> [B, nE, irrep_dim]
        v_perm = v_mod.permute(0, 2, 1, 3)  # [B, nE, N, irrep_dim]
        out_irrep = (attn.unsqueeze(-1) * v_perm).sum(dim=2)

        inv = invariant_features_from_irreps(out_irrep, self.out_irreps)  # [B, nE, inv_dim]
        return self.out_mlp(inv)
