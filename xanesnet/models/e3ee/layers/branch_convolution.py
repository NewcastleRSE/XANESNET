# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Invariant and equivariant convolution branches for the E3EE model.

Provides SchNet/PaiNN-style invariant convolution
(:class:`EnergyConditionedAtomConvolution`) and a NequIP/MACE-style equivariant
counterpart (:class:`EnergyConditionedEquivariantAtomConvolution`). Energy
conditioning is applied *after* aggregation to avoid materialising a per-edge
energy axis, which significantly reduces activation memory.
"""

from typing import cast

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct

from ..utils import invariant_feature_dim, invariant_features_from_irreps
from .basic import MLP, CosineCutoff, GaussianRBF, RadialMLP
from .branch_equivariant import EnergyIrrepModulation

# Reference: SchNet (Schuett et al., 2017), PaiNN (Schuett et al., 2021),
# NequIP (Batzner et al., 2022), MACE (Batatia et al., 2022).


class EnergyConditionedAtomConvolution(nn.Module):
    """
    Invariant SchNet/PaiNN-style convolution producing a per-(target-site, energy)
    latent.

    For each sample's target site::

        m_{target<-j} = MLP(h_j, z_j, RBF(dist), is_target_site)  # SchNet message
        m_{target<-j} *= cos_envelope(dist)
        [optional] m_{target<-j} *= sigmoid(MLP(h_target_site, h_j, RBF))  # PaiNN gate
        m_target = sum_j  (j in att-graph)                       # sum aggregation
        out_target[e] = MLP(EnergyMod_scalar(m_target, e_feat))   # energy mod

    Energy conditioning runs after aggregation, so per-edge tensors carry no
    ``nE`` axis (large activation-memory saving).

    Args:
        atom_dim: Dimension of invariant per-atom features.
        e_dim: Dimension of the energy RBF embedding.
        hidden_dim: Hidden dimension of all internal MLPs.
        latent_dim: Output (latent) dimension.
        att_cutoff: Radius of the attention neighborhood graph in Angstrom.
        rbf_dim: Number of Gaussian RBF bases for the target-site-to-atom distance.
        max_z: Maximum atomic number supported by the element embedding.
        z_emb_dim: Embedding dimension for atomic numbers.
        use_gate: If ``True``, apply a PaiNN-style learned scalar edge gate.
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
        use_gate: bool = True,
    ) -> None:
        """Initialize ``EnergyConditionedAtomConvolution``."""
        super().__init__()
        self.atom_dim = atom_dim
        self.e_dim = e_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rbf_dim = rbf_dim
        self.att_cutoff = float(att_cutoff)
        self.use_gate = bool(use_gate)

        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.dist_rbf = GaussianRBF(0.0, self.att_cutoff, rbf_dim)
        self.envelope = CosineCutoff(self.att_cutoff)

        message_in_dim = atom_dim + z_emb_dim + 1 + rbf_dim
        self.message_mlp = MLP(
            in_dim=message_in_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        if self.use_gate:
            gate_in_dim = atom_dim + atom_dim + rbf_dim + 1
            self.gate_mlp = MLP(
                in_dim=gate_in_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                n_layers=2,
            )

        self.energy_gate = MLP(
            in_dim=e_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        self.out_mlp = MLP(
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
        target_site_index: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Compute invariant convolution branch latent.

        Args:
            h: Invariant atom features, shape ``(B, N, H)``.
            z: Atomic numbers, shape ``(B, N)``.
            mask: Valid-atom mask, shape ``(B, N)``.
            e_feat: Energy RBF features, shape ``(nE, dE)``.
            target_site_index: Target-site index per sample, shape ``(B,)``.
            att_dst: Flat destination indices into ``B*N`` (target-site neighbors), shape ``(E_att,)``.
            att_dist: Target-site-to-atom distances in **Angstrom**, shape ``(E_att,)``.

        Returns:
            Latent tensor of shape ``(B, nE, latent_dim)``.
        """
        bsz, n_atoms, h_dim = h.shape
        n_energies = e_feat.shape[0]
        device = h.device
        dtype = h.dtype
        flat = bsz * n_atoms

        # Build per-atom (B, N) scope, distance, and target-site indicator.
        att_mask_flat = torch.zeros(flat, dtype=torch.bool, device=device)
        att_mask_flat[att_dst] = True
        att_dist_flat = torch.zeros(flat, dtype=dtype, device=device)
        att_dist_flat[att_dst] = att_dist.to(dtype=dtype)

        att_mask = att_mask_flat.view(bsz, n_atoms) & mask  # [B, N]
        att_dist_bn = att_dist_flat.view(bsz, n_atoms)
        rbf = self.dist_rbf(att_dist_bn)  # [B, N, rbf]

        zr = self.z_emb(z)  # [B, N, z_emb]
        batch_arange = torch.arange(bsz, device=device)
        is_target_site = torch.zeros(bsz, n_atoms, dtype=dtype, device=device)
        is_target_site[batch_arange, target_site_index] = 1.0

        # SchNet-style continuous-filter message per atom (energy-independent).
        msg_in = torch.cat([h, zr, is_target_site.unsqueeze(-1), rbf], dim=-1)
        msg = self.message_mlp(msg_in)  # [B, N, L]
        env = self.envelope(att_dist_bn).unsqueeze(-1)  # [B, N, 1]
        msg = msg * env

        # Optional PaiNN-style scalar edge gate.
        if self.use_gate:
            h_target_site = h[batch_arange, target_site_index, :]  # [B, H]
            h_target_site_bn = h_target_site.unsqueeze(1).expand(bsz, n_atoms, h_dim)
            gate_in = torch.cat([h_target_site_bn, h, rbf, is_target_site.unsqueeze(-1)], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_in))  # [B, N, 1]
            msg = msg * gate

        # Mask out atoms outside the attention scope.
        msg = msg * att_mask.unsqueeze(-1).to(dtype=dtype)

        # Pure sum aggregation -> per-target-site scalar feature.
        m_target_site = msg.sum(dim=1)  # [B, L]

        # Energy conditioning AFTER aggregation: only here does the nE axis
        # appear, and only on a per-target-site tensor.
        e_gate = self.energy_gate(e_feat)  # [nE, L]
        out = m_target_site.unsqueeze(1) * e_gate.unsqueeze(0)  # [B, nE, L]
        return self.out_mlp(out)


class EnergyConditionedEquivariantAtomConvolution(nn.Module):
    """
    NequIP/MACE-style equivariant convolution producing a per-(target-site,
    energy) latent.

    For each sample's target site::

        sh_j = Y(u_{target->j})
        v_{target<-j} = TP(h_full[j], sh_j; W(z_j, RBF, is_target_site))  # NequIP message
        v_{target<-j} *= cos_envelope(dist)
        [optional] v_{target<-j} *= sigmoid(MLP(h_target_site, h_j, RBF))  # PaiNN gate
        v_target = sum_j v_{target<-j}                              # sum agg
        v_{target,e} = EnergyIrrepModulation(v_target, e_feat)      # energy mod
        out = MLP(invariants(v_{target,e}))

    Args:
        atom_dim: Dimension of invariant per-atom features.
        irreps_node: Irreps of the full equivariant atom features.
        e_dim: Dimension of the energy RBF embedding.
        hidden_dim: Hidden dimension of all internal MLPs.
        latent_dim: Output (latent) dimension.
        att_cutoff: Radius of the attention neighborhood graph in Angstrom.
        attention_lmax: Maximum spherical-harmonics order for bond directions.
        attention_irreps: Target irreps of the equivariant message (e.g. ``"32x0e+16x1o"``).
        rbf_dim: Number of Gaussian RBF bases for the target-site-to-atom distance.
        max_z: Maximum atomic number supported by the element embedding.
        z_emb_dim: Embedding dimension for atomic numbers.
        use_gate: If ``True``, apply a PaiNN-style learned scalar edge gate.
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
        use_gate: bool = True,
    ) -> None:
        """Initialize ``EnergyConditionedEquivariantAtomConvolution``."""
        super().__init__()
        self.atom_dim = atom_dim
        self.e_dim = e_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rbf_dim = rbf_dim
        self.att_cutoff = float(att_cutoff)
        self.use_gate = bool(use_gate)

        self.sh_irreps = cast(o3.Irreps, o3.Irreps.spherical_harmonics(int(attention_lmax)))
        self.out_irreps = cast(o3.Irreps, o3.Irreps(attention_irreps))
        self.irreps_node = cast(o3.Irreps, o3.Irreps(irreps_node))

        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.dist_rbf = GaussianRBF(0.0, self.att_cutoff, rbf_dim)
        self.envelope = CosineCutoff(self.att_cutoff)

        # NequIP-style equivariant message.
        self.value_tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.sh_irreps,
            self.out_irreps,
            shared_weights=False,
        )
        weight_in_dim = z_emb_dim + 1 + rbf_dim
        self.value_weight_mlp = RadialMLP(weight_in_dim, hidden_dim, self.value_tp.weight_numel)

        if self.use_gate:
            gate_in_dim = atom_dim + atom_dim + rbf_dim + 1
            self.gate_mlp = MLP(
                in_dim=gate_in_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                n_layers=2,
            )

        self.energy_mod = EnergyIrrepModulation(self.out_irreps, e_dim=e_dim, hidden_dim=hidden_dim)

        self.inv_dim = invariant_feature_dim(self.out_irreps)
        self.out_mlp = MLP(
            in_dim=self.inv_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

    def forward(
        self,
        h: torch.Tensor,
        h_full: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        target_site_index: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
        att_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Apply equivariant convolution and return per-(target-site, energy) latent.

        Args:
            h: Invariant atom features ``(B, N, H)``.
            h_full: Full equivariant atom features ``(B, N, irreps_node.dim)``.
            z: Atomic numbers ``(B, N)``.
            mask: Valid-atom mask ``(B, N)``.
            e_feat: Energy RBF embedding ``(nE, e_dim)``.
            target_site_index: Target-site index per sample ``(B,)``.
            att_dst: Flat destination indices into ``B*N`` for the attention neighborhood ``(E_att,)``.
            att_dist: Target-site-to-atom distances ``(E_att,)``.
            att_vec: Target-site-to-atom displacement vectors ``(E_att, 3)``.

        Returns:
            Latent tensor of shape ``(B, nE, latent_dim)``.
        """
        bsz, n_atoms, h_dim = h.shape
        device = h.device
        dtype = h.dtype
        flat = bsz * n_atoms

        # Per-atom (B, N) scope / dist / direction (zeros outside scope).
        att_mask_flat = torch.zeros(flat, dtype=torch.bool, device=device)
        att_mask_flat[att_dst] = True
        att_dist_flat = torch.zeros(flat, dtype=dtype, device=device)
        att_dist_flat[att_dst] = att_dist.to(dtype=dtype)
        att_vec_flat = torch.zeros(flat, 3, dtype=dtype, device=device)
        att_vec_flat[att_dst] = att_vec.to(dtype=dtype)

        att_mask = att_mask_flat.view(bsz, n_atoms) & mask  # [B, N]

        eps_dist = att_dist_flat.clamp_min(1e-8)
        u = att_vec_flat / eps_dist.unsqueeze(-1)  # [flat, 3]
        sh = o3.spherical_harmonics(self.sh_irreps, u, normalize=True, normalization="component")
        rbf_flat = self.dist_rbf(att_dist_flat)

        zr = self.z_emb(z)
        zr_flat = zr.view(flat, -1)
        batch_arange = torch.arange(bsz, device=device)
        is_target_site = torch.zeros(bsz, n_atoms, dtype=dtype, device=device)
        is_target_site[batch_arange, target_site_index] = 1.0
        is_target_site_flat = is_target_site.view(flat, 1)

        # NequIP-style equivariant message (energy-independent).
        weight_in = torch.cat([zr_flat, is_target_site_flat, rbf_flat], dim=-1)
        tp_weights = self.value_weight_mlp(weight_in)
        h_full_flat = h_full.reshape(flat, self.irreps_node.dim)
        v_e = self.value_tp(h_full_flat, sh, tp_weights)  # [flat, D]

        # Smooth radial envelope.
        env = self.envelope(att_dist_flat).unsqueeze(-1)
        v_e = v_e * env

        # Optional PaiNN-style scalar gate (single scalar -> equivariance preserved).
        if self.use_gate:
            h_target_site = h[batch_arange, target_site_index, :]
            h_target_site_bn = h_target_site.unsqueeze(1).expand(bsz, n_atoms, h_dim)
            gate_in = torch.cat(
                [h_target_site_bn, h, rbf_flat.view(bsz, n_atoms, -1), is_target_site.unsqueeze(-1)],
                dim=-1,
            )
            gate = torch.sigmoid(self.gate_mlp(gate_in)).view(flat, 1)
            v_e = v_e * gate

        # Mask out atoms outside the attention scope.
        v_e = v_e * att_mask_flat.unsqueeze(-1).to(dtype=dtype)

        # Sum aggregation per sample (NequIP/MACE).
        v_bn = v_e.view(bsz, n_atoms, self.out_irreps.dim)
        v_target_site = v_bn.sum(dim=1)  # [B, D]

        # Energy conditioning AFTER aggregation -- the only nE-bearing tensor.
        v_mod = self.energy_mod(v_target_site, e_feat)  # [B, nE, D]
        inv = invariant_features_from_irreps(v_mod, self.out_irreps)  # [B, nE, inv_dim]
        return self.out_mlp(inv)
