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

from typing import cast

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct

from ..utils import invariant_feature_dim, invariant_features_from_irreps
from .basic import MLP, CosineCutoff, GaussianRBF, RadialMLP
from .branch_equivariant import EnergyIrrepModulation


class AllAtomAtomConvolution(nn.Module):
    """
    Invariant SchNet/PaiNN-style continuous-filter convolution branch.
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

        # Edge message: f(h_dst, z_dst, RBF, is_self) -> latent
        # (this is the SchNet "continuous-filter" message).
        message_in_dim = atom_dim + z_emb_dim + 1 + rbf_dim
        self.message_mlp = MLP(
            in_dim=message_in_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        # Optional PaiNN-style scalar gate: sigmoid(MLP(h_src, h_dst, RBF))
        # -> single scalar per edge, energy-independent.
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
        att_src: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
        absorber_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h:             [B, N, H] invariant atom features
            z:             [B, N] atomic numbers
            mask:          [B, N] valid-atom mask
            e_feat:        [nE, dE] energy feature embedding
            att_src:       [E_att] flat src indices (receivers) into B*N
            att_dst:       [E_att] flat dst indices (senders) into B*N
            att_dist:      [E_att] pair distances
            absorber_mask: optional [B, N]; if given, restrict receivers
                           to absorber sites.

        Returns:
            [B, N, nE, latent_dim]
        """
        bsz, n_atoms, h_dim = h.shape
        n_energies = e_feat.shape[0]
        device = h.device
        dtype = h.dtype
        flat = bsz * n_atoms

        h_flat = h.reshape(flat, h_dim)
        z_flat = z.reshape(flat)
        mask_flat = mask.reshape(flat)

        # Restrict edges to active receivers x valid senders.
        src_active = mask_flat.clone()
        if absorber_mask is not None:
            src_active = src_active & absorber_mask.reshape(flat)
        edge_active = src_active[att_src] & mask_flat[att_dst]

        ea_src = att_src[edge_active]
        ea_dst = att_dst[edge_active]
        ea_dist = att_dist[edge_active].to(dtype=dtype)
        n_edges = ea_src.shape[0]

        out_flat = torch.zeros(flat, self.latent_dim, device=device, dtype=dtype)

        if n_edges > 0:
            rbf = self.dist_rbf(ea_dist)  # [E, rbf]
            zr_d = self.z_emb(z_flat[ea_dst])  # [E, z_emb]
            is_self = (ea_src == ea_dst).to(dtype=dtype).unsqueeze(-1)
            h_dst = h_flat[ea_dst]  # [E, H]
            h_src = h_flat[ea_src]  # [E, H]

            # SchNet-style continuous-filter message.
            msg_in = torch.cat([h_dst, zr_d, is_self, rbf], dim=-1)
            msg = self.message_mlp(msg_in)  # [E, L]

            # Cosine envelope: guarantees radial continuity at att_cutoff.
            env = self.envelope(ea_dist).unsqueeze(-1)  # [E, 1]
            msg = msg * env

            # Optional PaiNN-style scalar edge gate.
            if self.use_gate:
                gate_in = torch.cat([h_src, h_dst, rbf, is_self], dim=-1)
                gate = torch.sigmoid(self.gate_mlp(gate_in))  # [E, 1]
                msg = msg * gate

            # Pure sum aggregation by receiver.
            out_flat.index_add_(0, ea_src, msg)

        # Energy conditioning is applied ONCE per receiver, AFTER aggregation.
        # ``out_flat`` has shape [flat, L]; broadcast-multiply with an
        # ``[nE, L]`` energy gate to materialize [flat, nE, L] only here.
        e_gate = self.energy_gate(e_feat)  # [nE, L]
        out = out_flat.unsqueeze(1) * e_gate.unsqueeze(0)  # [flat, nE, L]
        out = self.out_mlp(out)
        return out.view(bsz, n_atoms, n_energies, self.latent_dim)


class AllAtomEquivariantAtomConvolution(nn.Module):
    """
    NequIP/MACE-style equivariant convolution branch.

    For every active receiver atom (or absorber when ``use_absorber_mask``
    is enabled by the parent model):

        sh_{aj} = Y(u_{aj})              # spherical harmonics of bond dir
        v_{a<-j} = TP(h_full[j], sh_{aj}; W(z_j, RBF, is_self))
                                          # NequIP-style equivariant message
        v_{a<-j} *= cos_envelope(dist)   # smooth radial cutoff
        [optional] v_{a<-j} *= sigmoid(MLP(h_src, h_dst, RBF))  # PaiNN-style gate
        v_a = sum_j v_{a<-j}             # sum aggregation (NequIP/MACE)
        v_{a,e} = EnergyIrrepModulation(v_a, e_feat)  # per-receiver energy mod
        out = MLP(invariants(v_{a,e}))

    Self-edges (dist=0) are gated to ``Y_l = delta_{l,0}`` so they only
    contribute through the l=0 channel (cf. how the encoder handles
    self-loops). Memory: per-edge tensors are ``[E, D_irrep]`` -- no
    ``nE`` axis on edges. The only ``nE``-bearing tensor is the
    aggregated, per-receiver ``[N_recv, nE, D_irrep]``.
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

        # NequIP-style equivariant message: TP of dst irreps with SH of bond
        # direction, with TP weights from a radial / element / self MLP.
        self.value_tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.sh_irreps,
            self.out_irreps,
            shared_weights=False,
        )
        weight_in_dim = z_emb_dim + 1 + rbf_dim
        self.value_weight_mlp = RadialMLP(weight_in_dim, hidden_dim, self.value_tp.weight_numel)

        # Optional PaiNN-style scalar gate (energy-independent, single scalar
        # per edge). Multiplies ALL irrep components uniformly so equivariance
        # is preserved.
        if self.use_gate:
            gate_in_dim = atom_dim + atom_dim + rbf_dim + 1
            self.gate_mlp = MLP(
                in_dim=gate_in_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                n_layers=2,
            )

        # Per-receiver energy modulation of the aggregated equivariant feature.
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
        att_src: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
        att_vec: torch.Tensor,
        absorber_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args mirror ``AllAtomEquivariantAtomAttention.forward``. Returns
        ``[B, N, nE, latent_dim]``.
        """
        bsz, n_atoms, h_dim = h.shape
        n_energies = e_feat.shape[0]
        device = h.device
        dtype = h.dtype
        flat = bsz * n_atoms

        h_flat = h.reshape(flat, h_dim)
        h_full_flat = h_full.reshape(flat, self.irreps_node.dim)
        z_flat = z.reshape(flat)
        mask_flat = mask.reshape(flat)

        src_active = mask_flat.clone()
        if absorber_mask is not None:
            src_active = src_active & absorber_mask.reshape(flat)
        edge_active = src_active[att_src] & mask_flat[att_dst]

        ea_src = att_src[edge_active]
        ea_dst = att_dst[edge_active]
        ea_dist = att_dist[edge_active].to(dtype=dtype)
        ea_vec = att_vec[edge_active].to(dtype=dtype)
        n_edges = ea_src.shape[0]

        out_irrep = torch.zeros(flat, self.out_irreps.dim, device=device, dtype=dtype)

        if n_edges > 0:
            # Bond-direction unit vector; gate self-edges to l=0 only.
            eps_dist = ea_dist.clamp_min(1e-8)
            u = ea_vec / eps_dist.unsqueeze(-1)
            is_self = (ea_src == ea_dst).to(dtype=dtype).unsqueeze(-1)  # [E, 1]
            u_for_sh = torch.where(
                is_self.bool().expand_as(u),
                torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(u),
                u,
            )
            sh = o3.spherical_harmonics(self.sh_irreps, u_for_sh, normalize=True, normalization="component")
            sh = self._mask_l0_only(sh, is_self.squeeze(-1).bool())

            rbf = self.dist_rbf(ea_dist)  # [E, rbf]
            zr_d = self.z_emb(z_flat[ea_dst])  # [E, z_emb]
            h_dst = h_flat[ea_dst]
            h_src = h_flat[ea_src]
            h_full_dst = h_full_flat[ea_dst]

            # NequIP-style equivariant message.
            weight_in = torch.cat([zr_d, is_self, rbf], dim=-1)
            tp_weights = self.value_weight_mlp(weight_in)
            v_e = self.value_tp(h_full_dst, sh, tp_weights)  # [E, D]

            # Smooth radial envelope (continuity at att_cutoff).
            env = self.envelope(ea_dist).unsqueeze(-1)
            v_e = v_e * env

            # Optional PaiNN-style scalar gate. A single scalar multiplying
            # all irrep components preserves equivariance.
            if self.use_gate:
                gate_in = torch.cat([h_src, h_dst, rbf, is_self], dim=-1)
                gate = torch.sigmoid(self.gate_mlp(gate_in))  # [E, 1]
                v_e = v_e * gate

            # Sum aggregation by receiver (NequIP/MACE).
            out_irrep.index_add_(0, ea_src, v_e)

        # Energy conditioning ONCE per receiver, AFTER aggregation.
        v_mod = self.energy_mod(out_irrep, e_feat)  # [flat, nE, D]
        inv = invariant_features_from_irreps(v_mod, self.out_irreps)  # [flat, nE, inv_dim]
        out = self.out_mlp(inv)
        return out.view(bsz, n_atoms, n_energies, self.latent_dim)

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
        return out
