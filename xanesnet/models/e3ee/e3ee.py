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
    AbsorberPathAggregator,
    EnergyConditionedAbsorberBranch,
    EnergyConditionedAtomAttention,
    EnergyConditionedAtomConvolution,
    EnergyConditionedEquivariantAbsorberHead,
    EnergyConditionedEquivariantAtomAttention,
    EnergyConditionedEquivariantAtomConvolution,
    EnergyRBFEmbedding,
    EquivariantAtomEncoder,
    PairElementEnergyScattering,
)
from .utils import invariant_feature_dim, invariant_features_from_irreps


@ModelRegistry.register("e3ee")
class E3EE(Model):
    """
    Absorber-centred energy-embedded E3-equivariant XANES model.

    Architecture:
    - Equivariant atom encoder (e3nn spherical harmonics message passing)
    - Invariant atomwise summaries from all irreps
    - Energy-conditioned attention over atoms
    - Late equivariant absorber head
    - Optional 3-body path branch
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

        # Branch 1 (optional): absorber invariant features + energy
        if self.use_invariant_branch:
            self.abs_branch = EnergyConditionedAbsorberBranch(
                atom_dim=self._inv_dim,
                e_dim=energy_rbf_dim,
                hidden_dim=head_hidden_dim,
                out_dim=latent_dim,
            )

        # Branch 2 (optional): energy-conditioned atom attention
        if self.use_attention_branch:
            self.atom_attention = EnergyConditionedAtomAttention(
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

        # Branch 2b (optional): equivariant counterpart of the attention branch
        if self.use_eq_attention_branch:
            self.eq_atom_attention = EnergyConditionedEquivariantAtomAttention(
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

        # Branch 3 (optional): late equivariant absorber head
        if self.use_equivariant_branch:
            self.eq_abs_head = EnergyConditionedEquivariantAbsorberHead(
                irreps_node=self.atom_encoder.irreps_node,
                e_dim=energy_rbf_dim,
                hidden_dim=head_hidden_dim,
                out_dim=latent_dim,
            )

        # Branch 2c (optional): SchNet/PaiNN-style invariant convolution.
        if self.use_conv_branch:
            self.atom_convolution = EnergyConditionedAtomConvolution(
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
            self.eq_atom_convolution = EnergyConditionedEquivariantAtomConvolution(
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

        # Branch 4 (optional): 3-body path terms
        if self.use_path_branch:
            self.pair_elem_energy = PairElementEnergyScattering(
                max_z=max_z,
                z_emb_dim=32,
                e_dim=energy_rbf_dim,
                hidden_dim=128,
                out_dim=scatter_dim,
            )
            self.path_agg = AbsorberPathAggregator(
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
        absorber_index: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
        att_vec: torch.Tensor,
        energies: torch.Tensor,
        path_j: torch.Tensor,
        path_k: torch.Tensor,
        path_r0j: torch.Tensor,
        path_r0k: torch.Tensor,
        path_rjk: torch.Tensor,
        path_cosangle: torch.Tensor,
        path_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:              [B, N]    atomic numbers (int64, padded)
            mask:           [B, N]    valid-atom mask
            absorber_index: [B]       absorber atom index per sample (padded layout)
            edge_src:       [E]       flat source indices into B*N
            edge_dst:       [E]       flat destination indices into B*N
            edge_weight:    [E]       edge lengths
            edge_vec:       [E, 3]    edge displacement vectors
            energies:       [B, nE]   energy grid (unused directly; nE drives RBF)
            path_j:         [P]       flat j index into B*N
            path_k:         [P]       flat k index into B*N
            path_r0j:       [P]
            path_r0k:       [P]
            path_rjk:       [P]
            path_cosangle:  [P]
            path_batch:     [P]       batch index per path

        Returns:
            [B, nE] predicted XANES intensities
        """
        bsz, n_atoms = x.shape
        device = x.device
        dtype = edge_vec.dtype

        n_energies = energies.shape[-1]  # nE
        energy_indices = torch.arange(n_energies, device=device, dtype=dtype)
        e_feat = self.energy_embedding(energy_indices)  # [nE, energy_rbf_dim]

        h_full = self.atom_encoder(
            z=x,
            mask=mask,
            absorber_index=absorber_index,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_weight=edge_weight,
            edge_vec=edge_vec,
        )  # [B, N, irreps_dim]

        h = invariant_features_from_irreps(h_full, self.atom_encoder.irreps_node)  # [B, N, inv_dim]

        batch_arange = torch.arange(bsz, device=device)
        h_abs = h[batch_arange, absorber_index, :]
        h_abs_full = h_full[batch_arange, absorber_index, :]

        parts = []

        # Branch 1 (optional): absorber invariant features + energy
        if self.use_invariant_branch:
            abs_lat = self.abs_branch(h_abs, e_feat)  # [B, nE, latent]
            parts.append(abs_lat)

        # Branch 2 (optional): energy-conditioned atom attention
        if self.use_attention_branch:
            attn_lat = self.atom_attention(
                h=h,
                z=x,
                mask=mask,
                e_feat=e_feat,
                absorber_index=absorber_index,
                att_dst=att_dst,
                att_dist=att_dist,
            )  # [B, nE, latent]
            parts.append(attn_lat)

        # Branch 2b (optional): equivariant atom attention
        if self.use_eq_attention_branch:
            eq_attn_lat = self.eq_atom_attention(
                h=h,
                h_full=h_full,
                z=x,
                mask=mask,
                e_feat=e_feat,
                absorber_index=absorber_index,
                att_dst=att_dst,
                att_dist=att_dist,
                att_vec=att_vec,
            )  # [B, nE, latent]
            parts.append(eq_attn_lat)

        # Branch 3 (optional): late equivariant absorber head
        if self.use_equivariant_branch:
            eq_abs_lat = self.eq_abs_head(h_abs_full, e_feat)  # [B, nE, latent]
            parts.append(eq_abs_lat)

        # Branch 2c (optional): invariant convolution.
        if self.use_conv_branch:
            conv_lat = self.atom_convolution(
                h=h,
                z=x,
                mask=mask,
                e_feat=e_feat,
                absorber_index=absorber_index,
                att_dst=att_dst,
                att_dist=att_dist,
            )  # [B, nE, latent]
            parts.append(conv_lat)

        # Branch 2d (optional): equivariant convolution.
        if self.use_eq_conv_branch:
            eq_conv_lat = self.eq_atom_convolution(
                h=h,
                h_full=h_full,
                z=x,
                mask=mask,
                e_feat=e_feat,
                absorber_index=absorber_index,
                att_dst=att_dst,
                att_dist=att_dist,
                att_vec=att_vec,
            )  # [B, nE, latent]
            parts.append(eq_conv_lat)

        # Branch 4 (optional): 3-body path terms
        if self.use_path_branch:
            h_flat = h.reshape(bsz * n_atoms, self._inv_dim)
            z_flat = x.reshape(bsz * n_atoms)
            path_lat = self.path_agg(
                h_flat=h_flat,
                z_flat=z_flat,
                pair_elem_energy=self.pair_elem_energy,
                e_feat=e_feat,
                path_j=path_j,
                path_k=path_k,
                path_r0j=path_r0j,
                path_r0k=path_r0k,
                path_rjk=path_rjk,
                path_cosangle=path_cosangle,
                path_batch=path_batch,
                bsz=bsz,
            )  # [B, nE, latent]
            parts.append(path_lat)

        combined = torch.cat(parts, dim=-1)  # [B, nE, head_in_dim]
        out = self.head(combined).squeeze(-1)  # [B, nE]
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
