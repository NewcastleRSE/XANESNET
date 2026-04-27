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

from xanesnet.components import BiasInitRegistry, WeightInitRegistry
from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry
from .layers import (
    MLP,
    AllAtomAtomAttention,
    AllAtomAtomConvolution,
    AllAtomEnergyBranch,
    AllAtomEquivariantAtomAttention,
    AllAtomEquivariantAtomConvolution,
    AllAtomEquivariantHead,
    AllAtomPathAggregator,
    EnergyRBFEmbedding,
    EquivariantAtomEncoder,
    PairElementEnergyScattering,
)
from .utils import invariant_feature_dim, invariant_features_from_irreps


@ModelRegistry.register("e3ee_full")
class E3EEFull(Model):
    """
    Multi-Absorber E3-equivariant XANES model.
    """

    def __init__(
        self,
        model_type: str,
        # params:
        out_size: int,
        max_z: int,
        atom_emb_dim: int,
        atom_hidden_dim: int,
        atom_layers: int,
        local_cutoff: float,
        rbf_dim: int,
        energy_rbf_dim: int,
        scatter_dim: int,
        latent_dim: int,
        head_hidden_dim: int,
        e3nn_irreps: str,
        e3nn_irreps_message: str,
        e3nn_lmax: int,
        out_mlp_layers: int,
        use_invariant_branch: bool,
        use_attention_branch: bool,
        use_equivariant_branch: bool,
        use_eq_attention_branch: bool,
        use_conv_branch: bool,
        use_eq_conv_branch: bool,
        use_path_branch: bool,
        use_absorber_mask: bool,
        residual_scale_init: float,
        attention_heads: int,
        attention_rbf_dim: int,
        attention_lmax: int,
        attention_irreps: str,
        att_cutoff: float,
        conv_use_gate: bool = True,
    ) -> None:
        super().__init__(model_type)

        self.out_size = out_size
        self.max_z = max_z
        self.atom_emb_dim = atom_emb_dim
        self.atom_hidden_dim = atom_hidden_dim
        self.atom_layers = atom_layers
        self.local_cutoff = local_cutoff
        self.rbf_dim = rbf_dim
        self.energy_rbf_dim = energy_rbf_dim
        self.scatter_dim = scatter_dim
        self.latent_dim = latent_dim
        self.head_hidden_dim = head_hidden_dim
        self.e3nn_irreps = e3nn_irreps
        self.e3nn_irreps_message = e3nn_irreps_message
        self.e3nn_lmax = e3nn_lmax
        self.out_mlp_layers = out_mlp_layers
        self.use_invariant_branch = use_invariant_branch
        self.use_attention_branch = use_attention_branch
        self.use_equivariant_branch = use_equivariant_branch
        self.use_eq_attention_branch = use_eq_attention_branch
        self.use_conv_branch = use_conv_branch
        self.use_eq_conv_branch = use_eq_conv_branch
        self.use_path_branch = use_path_branch
        self.use_absorber_mask = use_absorber_mask
        self.residual_scale_init = residual_scale_init
        self.attention_heads = attention_heads
        self.attention_rbf_dim = attention_rbf_dim
        self.attention_lmax = attention_lmax
        self.attention_irreps = attention_irreps
        self.att_cutoff = att_cutoff
        self.conv_use_gate = conv_use_gate

        # Energy index range for RBF embedding (uses grid indices 0..out_size-1)
        self._energy_min = 0.0
        self._energy_max = float(out_size - 1)

        # Equivariant atom encoder
        self.atom_encoder = EquivariantAtomEncoder(
            max_z=max_z,
            cutoff=local_cutoff,
            num_interactions=atom_layers,
            rbf_dim=rbf_dim,
            lmax=e3nn_lmax,
            node_attr_dim=atom_emb_dim,
            hidden_dim=atom_hidden_dim,
            irreps_node=e3nn_irreps,
            irreps_message=e3nn_irreps_message,
            residual_scale_init=residual_scale_init,
        )

        self._inv_dim = invariant_feature_dim(self.atom_encoder.irreps_node)

        # Energy embedding
        self.energy_embedding = EnergyRBFEmbedding(
            e_min=self._energy_min,
            e_max=self._energy_max,
            n_rbf=energy_rbf_dim,
        )

        # Branch 1 (optional): per-atom invariant features + energy
        if self.use_invariant_branch:
            self.abs_branch = AllAtomEnergyBranch(
                atom_dim=self._inv_dim,
                e_dim=energy_rbf_dim,
                hidden_dim=head_hidden_dim,
                out_dim=latent_dim,
            )

        # Branch 2 (optional): energy-conditioned atom attention (per-atom query, sparse).
        if self.use_attention_branch:
            self.atom_attention = AllAtomAtomAttention(
                atom_dim=self._inv_dim,
                e_dim=energy_rbf_dim,
                hidden_dim=atom_hidden_dim,
                latent_dim=latent_dim,
                att_cutoff=att_cutoff,
                rbf_dim=attention_rbf_dim,
                max_z=max_z,
                z_emb_dim=32,
                n_heads=attention_heads,
            )

        # Branch 2b (optional): equivariant per-atom attention
        if self.use_eq_attention_branch:
            self.eq_atom_attention = AllAtomEquivariantAtomAttention(
                atom_dim=self._inv_dim,
                irreps_node=self.atom_encoder.irreps_node,
                e_dim=energy_rbf_dim,
                hidden_dim=atom_hidden_dim,
                latent_dim=latent_dim,
                att_cutoff=att_cutoff,
                attention_lmax=attention_lmax,
                attention_irreps=attention_irreps,
                rbf_dim=attention_rbf_dim,
                max_z=max_z,
                z_emb_dim=32,
                n_heads=attention_heads,
            )

        # Branch 2c (optional): SchNet/PaiNN-style invariant convolution.
        if self.use_conv_branch:
            self.atom_convolution = AllAtomAtomConvolution(
                atom_dim=self._inv_dim,
                e_dim=energy_rbf_dim,
                hidden_dim=atom_hidden_dim,
                latent_dim=latent_dim,
                att_cutoff=att_cutoff,
                rbf_dim=attention_rbf_dim,
                max_z=max_z,
                z_emb_dim=32,
                use_gate=conv_use_gate,
            )

        # Branch 2d (optional): NequIP/MACE-style equivariant convolution.
        if self.use_eq_conv_branch:
            self.eq_atom_convolution = AllAtomEquivariantAtomConvolution(
                atom_dim=self._inv_dim,
                irreps_node=self.atom_encoder.irreps_node,
                e_dim=energy_rbf_dim,
                hidden_dim=atom_hidden_dim,
                latent_dim=latent_dim,
                att_cutoff=att_cutoff,
                attention_lmax=attention_lmax,
                attention_irreps=attention_irreps,
                rbf_dim=attention_rbf_dim,
                max_z=max_z,
                z_emb_dim=32,
                use_gate=conv_use_gate,
            )

        # Branch 3 (optional): late equivariant head, per atom
        if self.use_equivariant_branch:
            self.eq_head = AllAtomEquivariantHead(
                irreps_node=self.atom_encoder.irreps_node,
                e_dim=energy_rbf_dim,
                hidden_dim=head_hidden_dim,
                out_dim=latent_dim,
            )

        # Branch 4 (optional): 3-body path terms, per site
        if self.use_path_branch:
            self.pair_elem_energy = PairElementEnergyScattering(
                max_z=max_z,
                z_emb_dim=32,
                e_dim=energy_rbf_dim,
                hidden_dim=128,
                out_dim=scatter_dim,
            )
            self.path_agg = AllAtomPathAggregator(
                atom_dim=self._inv_dim,
                rbf_dim=rbf_dim,
                geom_hidden_dim=128,
                scatter_dim=scatter_dim,
                out_dim=latent_dim,
                cutoff=local_cutoff,
            )

        # Final head MLP
        n_active = (
            int(self.use_invariant_branch)
            + int(self.use_attention_branch)
            + int(self.use_equivariant_branch)
            + int(self.use_eq_attention_branch)
            + int(self.use_conv_branch)
            + int(self.use_eq_conv_branch)
            + int(self.use_path_branch)
        )
        head_in_dim = n_active * latent_dim
        self.head = MLP(
            in_dim=head_in_dim,
            hidden_dim=head_hidden_dim,
            out_dim=1,
            n_layers=out_mlp_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        absorber_mask: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
        att_src: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
        att_vec: torch.Tensor,
        energies: torch.Tensor,
        path_center: torch.Tensor,
        path_j: torch.Tensor,
        path_k: torch.Tensor,
        path_r0j: torch.Tensor,
        path_r0k: torch.Tensor,
        path_rjk: torch.Tensor,
        path_cosangle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass producing a spectrum for every atom.

        Args:
            x:             [B, N]   atomic numbers (int64, padded)
            mask:          [B, N]   valid-atom mask
            edge_src:      [E]      flat source indices into B*N
            edge_dst:      [E]      flat destination indices into B*N
            edge_weight:   [E]      edge lengths
            edge_vec:      [E, 3]   edge displacement vectors
            energies:      [n_abs, nE] energy grid (only ``nE`` is used here)
            path_center:   [P]      flat site index per path (into B*N)
            path_j:        [P]      flat j atom index (into B*N)
            path_k:        [P]      flat k atom index (into B*N)
            path_r0j:      [P]
            path_r0k:      [P]
            path_rjk:      [P]
            path_cosangle: [P]

        Returns:
            [B, N, nE] predicted XANES intensities for every atom (padded).
            The caller is expected to mask out atoms without ground truth.
        """
        bsz, n_atoms = x.shape
        device = x.device
        dtype = edge_vec.dtype

        n_energies = energies.shape[-1]
        energy_indices = torch.arange(n_energies, device=device, dtype=dtype)
        e_feat = self.energy_embedding(energy_indices)  # [nE, energy_rbf_dim]

        h_full = self.atom_encoder(
            z=x,
            mask=mask,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_weight=edge_weight,
            edge_vec=edge_vec,
        )  # [B, N, irreps_dim]

        h = invariant_features_from_irreps(h_full, self.atom_encoder.irreps_node)  # [B, N, inv_dim]

        active_mask = absorber_mask if self.use_absorber_mask else None

        parts = []

        # Branch 1
        if self.use_invariant_branch:
            abs_lat = self.abs_branch(h, e_feat)  # [B, N, nE, latent]
            parts.append(abs_lat)

        # Branch 2a
        if self.use_attention_branch:
            attn_lat = self.atom_attention(
                h=h,
                z=x,
                mask=mask,
                e_feat=e_feat,
                att_src=att_src,
                att_dst=att_dst,
                att_dist=att_dist,
                absorber_mask=active_mask,
            )  # [B, N, nE, latent]
            parts.append(attn_lat)

        # Branch 2b
        if self.use_eq_attention_branch:
            eq_attn_lat = self.eq_atom_attention(
                h=h,
                h_full=h_full,
                z=x,
                mask=mask,
                e_feat=e_feat,
                att_src=att_src,
                att_dst=att_dst,
                att_dist=att_dist,
                att_vec=att_vec,
                absorber_mask=active_mask,
            )  # [B, N, nE, latent]
            parts.append(eq_attn_lat)

        # Branch 2c
        if self.use_conv_branch:
            conv_lat = self.atom_convolution(
                h=h,
                z=x,
                mask=mask,
                e_feat=e_feat,
                att_src=att_src,
                att_dst=att_dst,
                att_dist=att_dist,
                absorber_mask=active_mask,
            )  # [B, N, nE, latent]
            parts.append(conv_lat)

        # Branch 2d
        if self.use_eq_conv_branch:
            eq_conv_lat = self.eq_atom_convolution(
                h=h,
                h_full=h_full,
                z=x,
                mask=mask,
                e_feat=e_feat,
                att_src=att_src,
                att_dst=att_dst,
                att_dist=att_dist,
                att_vec=att_vec,
                absorber_mask=active_mask,
            )  # [B, N, nE, latent]
            parts.append(eq_conv_lat)

        # Branch 3
        if self.use_equivariant_branch:
            eq_lat = self.eq_head(h_full, e_feat)  # [B, N, nE, latent]
            parts.append(eq_lat)

        # Branch 4
        if self.use_path_branch:
            h_flat = h.reshape(bsz * n_atoms, self._inv_dim)
            z_flat = x.reshape(bsz * n_atoms)

            if active_mask is not None:
                abs_flat = active_mask.reshape(bsz * n_atoms)
                keep = abs_flat[path_center]
                p_center = path_center[keep]
                p_j = path_j[keep]
                p_k = path_k[keep]
                p_r0j = path_r0j[keep]
                p_r0k = path_r0k[keep]
                p_rjk = path_rjk[keep]
                p_cos = path_cosangle[keep]
            else:
                p_center, p_j, p_k = path_center, path_j, path_k
                p_r0j, p_r0k = path_r0j, path_r0k
                p_rjk, p_cos = path_rjk, path_cosangle

            path_lat = self.path_agg(
                h_flat=h_flat,
                z_flat=z_flat,
                pair_elem_energy=self.pair_elem_energy,
                e_feat=e_feat,
                path_center=p_center,
                path_j=p_j,
                path_k=p_k,
                path_r0j=p_r0j,
                path_r0k=p_r0k,
                path_rjk=p_rjk,
                path_cosangle=p_cos,
                bsz=bsz,
                n_atoms=n_atoms,
            )  # [B, N, nE, latent]
            parts.append(path_lat)

        combined = torch.cat(parts, dim=-1)  # [B, N, nE, head_in_dim]
        out = self.head(combined).squeeze(-1)  # [B, N, nE]

        if self.use_absorber_mask:
            out = out * absorber_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        weight_init_fn = WeightInitRegistry.get(weights_init, **kwargs)
        bias_init_fn = BiasInitRegistry.get(bias_init)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight_init_fn(module.weight)
                if module.bias is not None:
                    bias_init_fn(module.bias)
            elif isinstance(module, nn.Embedding):
                module.reset_parameters()

    @property
    def signature(self) -> Config:
        """
        Return model signature as a configuration dictionary.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "out_size": self.out_size,
                "max_z": self.max_z,
                "atom_emb_dim": self.atom_emb_dim,
                "atom_hidden_dim": self.atom_hidden_dim,
                "atom_layers": self.atom_layers,
                "local_cutoff": self.local_cutoff,
                "rbf_dim": self.rbf_dim,
                "energy_rbf_dim": self.energy_rbf_dim,
                "scatter_dim": self.scatter_dim,
                "latent_dim": self.latent_dim,
                "head_hidden_dim": self.head_hidden_dim,
                "e3nn_irreps": self.e3nn_irreps,
                "e3nn_irreps_message": self.e3nn_irreps_message,
                "e3nn_lmax": self.e3nn_lmax,
                "out_mlp_layers": self.out_mlp_layers,
                "use_invariant_branch": self.use_invariant_branch,
                "use_attention_branch": self.use_attention_branch,
                "use_equivariant_branch": self.use_equivariant_branch,
                "use_eq_attention_branch": self.use_eq_attention_branch,
                "use_conv_branch": self.use_conv_branch,
                "use_eq_conv_branch": self.use_eq_conv_branch,
                "use_path_branch": self.use_path_branch,
                "use_absorber_mask": self.use_absorber_mask,
                "residual_scale_init": self.residual_scale_init,
                "attention_heads": self.attention_heads,
                "attention_rbf_dim": self.attention_rbf_dim,
                "attention_lmax": self.attention_lmax,
                "attention_irreps": self.attention_irreps,
                "att_cutoff": self.att_cutoff,
                "conv_use_gate": self.conv_use_gate,
            }
        )
        return signature
