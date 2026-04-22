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
    AllAtomEnergyBranch,
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
    Absorber-agnostic E3-equivariant XANES model predicting a spectrum for
    every atom in the structure.

    Architecture mirrors ``E3EE`` but the encoder has no absorber flag and
    every branch produces a ``[B, N, nE, ...]`` tensor so that spectra are
    emitted natively for all sites. The training loop uses an ``absorber_mask``
    (identical pattern to SchNet / DimeNet) to select sites with ground truth.
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
        use_path_branch: bool,
        residual_scale_init: float,
        attention_heads: int,
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
        self.use_path_branch = use_path_branch
        self.residual_scale_init = residual_scale_init
        self.attention_heads = attention_heads

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

        # Branch 2 (optional): energy-conditioned atom attention (per-atom query, SDPA)
        if self.use_attention_branch:
            self.atom_attention = AllAtomAtomAttention(
                atom_dim=self._inv_dim,
                e_dim=energy_rbf_dim,
                hidden_dim=atom_hidden_dim,
                latent_dim=latent_dim,
                max_z=max_z,
                z_emb_dim=32,
                n_heads=attention_heads,
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
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
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

        parts = []

        if self.use_invariant_branch:
            abs_lat = self.abs_branch(h, e_feat)  # [B, N, nE, latent]
            parts.append(abs_lat)

        if self.use_attention_branch:
            attn_lat = self.atom_attention(h=h, z=x, mask=mask, e_feat=e_feat)  # [B, N, nE, latent]
            parts.append(attn_lat)

        if self.use_equivariant_branch:
            eq_lat = self.eq_head(h_full, e_feat)  # [B, N, nE, latent]
            parts.append(eq_lat)

        if self.use_path_branch:
            h_flat = h.reshape(bsz * n_atoms, self._inv_dim)
            z_flat = x.reshape(bsz * n_atoms)
            path_lat = self.path_agg(
                h_flat=h_flat,
                z_flat=z_flat,
                pair_elem_energy=self.pair_elem_energy,
                e_feat=e_feat,
                path_center=path_center,
                path_j=path_j,
                path_k=path_k,
                path_r0j=path_r0j,
                path_r0k=path_r0k,
                path_rjk=path_rjk,
                path_cosangle=path_cosangle,
                bsz=bsz,
                n_atoms=n_atoms,
            )  # [B, N, nE, latent]
            parts.append(path_lat)

        combined = torch.cat(parts, dim=-1)  # [B, N, nE, head_in_dim]
        out = self.head(combined).squeeze(-1)  # [B, N, nE]
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
                "use_path_branch": self.use_path_branch,
                "residual_scale_init": self.residual_scale_init,
                "attention_heads": self.attention_heads,
            }
        )
        return signature
