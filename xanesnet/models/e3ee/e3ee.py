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
    EnergyConditionedEquivariantAbsorberHead,
    EnergyRBFEmbedding,
    EquivariantAtomEncoder,
    PairElementEnergyScattering,
    build_absorber_relative_geometry,
    invariant_feature_dim,
    invariant_features_from_irreps,
)


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
        use_path_terms: bool,
        max_paths_per_structure: int,
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
        self.use_path_terms = use_path_terms
        self.max_paths_per_structure = max_paths_per_structure
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

        # Branch 1: absorber invariant features + energy
        self.abs_branch = EnergyConditionedAbsorberBranch(
            atom_dim=self._inv_dim,
            e_dim=energy_rbf_dim,
            hidden_dim=head_hidden_dim,
            out_dim=latent_dim,
        )

        # Branch 2: energy-conditioned atom attention
        self.atom_attention = EnergyConditionedAtomAttention(
            atom_dim=self._inv_dim,
            e_dim=energy_rbf_dim,
            rbf_dim=rbf_dim,
            hidden_dim=atom_hidden_dim,
            latent_dim=latent_dim,
            cutoff=local_cutoff,
            max_z=max_z,
            z_emb_dim=32,
            n_heads=attention_heads,
        )

        # Branch 3: late equivariant absorber head
        self.eq_abs_head = EnergyConditionedEquivariantAbsorberHead(
            irreps_node=self.atom_encoder.irreps_node,
            e_dim=energy_rbf_dim,
            hidden_dim=head_hidden_dim,
            out_dim=latent_dim,
        )

        # Branch 4 (optional): 3-body path terms
        if self.use_path_terms:
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
                max_paths_per_structure=max_paths_per_structure,
            )

        # Final head MLP
        head_in_dim = 4 * latent_dim if self.use_path_terms else 3 * latent_dim
        self.head = MLP(
            in_dim=head_in_dim,
            hidden_dim=head_hidden_dim,
            out_dim=1,
            n_layers=out_mlp_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:        [B, N]    atomic numbers (int64)
            pos:      [B, N, 3] Cartesian coordinates (float32)
            mask:     [B, N]    boolean mask indicating valid atoms (True) vs padding (False)
            energies: [B, nE]   energy grid values (float32)

        Returns:
            [B, nE] predicted XANES intensities
        """
        B, N = x.shape
        device = pos.device

        # Mask: all atoms are valid (no padding in E3EE dataset)
        absorber_index = 0

        # Cache absorber-relative geometry
        geom = build_absorber_relative_geometry(pos=pos, mask=mask, absorber_index=absorber_index)

        # Energy grid as integer indices for RBF embedding
        nE = energies.shape[-1]
        energy_indices = torch.arange(nE, device=device, dtype=pos.dtype)
        e_feat = self.energy_embedding(energy_indices)  # [nE, energy_rbf_dim]

        # Equivariant atom encoding
        h_full = self.atom_encoder(
            z=x, pos=pos, mask=mask, absorber_index=absorber_index, geom=geom
        )  # [B, N, irreps_dim]

        # Extract invariant features
        h = invariant_features_from_irreps(h_full, self.atom_encoder.irreps_node)  # [B, N, inv_dim]

        # Branch 1: absorber invariant + energy
        # TODO interesting idea to add energies here
        abs_lat = self.abs_branch(h[:, absorber_index, :], e_feat)  # [B, nE, latent]

        # Branch 2: energy-conditioned attention
        attn_lat = self.atom_attention(
            h=h,
            z=x,
            pos=pos,
            mask=mask,
            e_feat=e_feat,
            absorber_index=absorber_index,
            geom=geom,
        )  # [B, nE, latent]

        # Branch 3: late equivariant absorber head
        eq_abs_lat = self.eq_abs_head(h_full[:, absorber_index, :], e_feat)  # [B, nE, latent]

        parts = [abs_lat, attn_lat, eq_abs_lat]

        # Branch 4 (optional): path terms
        if self.use_path_terms:
            path_lat = self.path_agg(
                h=h,
                z=x,
                pos=pos,
                mask=mask,
                pair_elem_energy=self.pair_elem_energy,
                e_feat=e_feat,
                absorber_index=absorber_index,
                geom=geom,
            )  # [B, nE, latent]
            parts.append(path_lat)

        combined = torch.cat(parts, dim=-1)  # [B, nE, head_in_dim]
        out = self.head(combined).squeeze(-1)  # [B, nE]
        return out

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """
        Initialize model weights and biases.

        Applies the given initialization to all ``nn.Linear`` layers.
        e3nn-specific layers (``o3.Linear``, ``FullyConnectedTensorProduct``)
        keep their default initialization. Embedding layers are reset to
        default.
        """
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
                "use_path_terms": self.use_path_terms,
                "max_paths_per_structure": self.max_paths_per_structure,
                "residual_scale_init": self.residual_scale_init,
                "attention_heads": self.attention_heads,
            }
        )
        return signature
