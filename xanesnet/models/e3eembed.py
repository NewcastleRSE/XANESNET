"""
Absorber-centred e3ee XANES model for XANESNET,
with neighbour and optional path scattering branches.
"""

from typing import Optional, List

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct

from xanesnet.models.base_model import Model
from xanesnet.registry import register_model, register_scheme


# ============================================================
# Utilities
# ============================================================

def build_absorber_relative_geometry(
    z: torch.Tensor,
    pos: torch.Tensor,
    mask: torch.Tensor,
    absorber_index: int = 0,
):
    abs_pos = pos[:, absorber_index, :].unsqueeze(1)
    rel = pos - abs_pos
    r = torch.linalg.norm(rel, dim=-1)
    u = rel / r.unsqueeze(-1).clamp_min(1e-8)

    valid_neigh = mask.clone()
    valid_neigh[:, absorber_index] = False

    return {
        "rel": rel,
        "r": r,
        "u": u,
        "valid_neigh": valid_neigh,
    }


class GaussianRBF(nn.Module):
    def __init__(self, start: float, stop: float, n_rbf: int, gamma: Optional[float] = None):
        super().__init__()
        centers = torch.linspace(start, stop, n_rbf)
        self.register_buffer("centers", centers)
        if gamma is None:
            delta = (stop - start) / max(n_rbf - 1, 1)
            gamma = 1.0 / (delta * delta + 1e-12)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)


class EnergyRBFEmbedding(nn.Module):
    def __init__(self, e_min: float, e_max: float, n_rbf: int):
        super().__init__()
        self.rbf = GaussianRBF(e_min, e_max, n_rbf)

    def forward(self, energies: torch.Tensor) -> torch.Tensor:
        return self.rbf(energies)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RadialMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# e3ee graph builder
# ============================================================

class RadiusGraphBuilder:
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, pos: torch.Tensor, mask: torch.Tensor):
        device = pos.device
        B, N, _ = pos.shape

        edge_src_all = []
        edge_dst_all = []
        edge_vec_all = []

        for b in range(B):
            valid = torch.where(mask[b])[0]
            if valid.numel() == 0:
                continue

            pos_b = pos[b, valid]
            diff = pos_b[:, None, :] - pos_b[None, :, :]
            dist = torch.linalg.norm(diff, dim=-1)

            edge_mask = (dist <= self.cutoff) & (dist > 1e-8)
            src_local, dst_local = torch.where(edge_mask)

            offset = b * N
            src = valid[src_local] + offset
            dst = valid[dst_local] + offset

            edge_src_all.append(src)
            edge_dst_all.append(dst)
            edge_vec_all.append(pos[b, valid[dst_local]] - pos[b, valid[src_local]])

        if len(edge_src_all) == 0:
            edge_src = torch.zeros(0, dtype=torch.long, device=device)
            edge_dst = torch.zeros(0, dtype=torch.long, device=device)
            edge_vec = torch.zeros(0, 3, dtype=pos.dtype, device=device)
        else:
            edge_src = torch.cat(edge_src_all, dim=0)
            edge_dst = torch.cat(edge_dst_all, dim=0)
            edge_vec = torch.cat(edge_vec_all, dim=0)

        return edge_src, edge_dst, edge_vec


def scalar_indices_from_irreps(irreps: o3.Irreps) -> List[int]:
    idx = []
    offset = 0
    for mul, ir in irreps:
        block_dim = mul * ir.dim
        if ir.l == 0:
            idx.extend(range(offset, offset + block_dim))
        offset += block_dim
    return idx


# ============================================================
# e3ee encoder
# ============================================================

class EquivariantInteractionBlock(nn.Module):
    def __init__(
        self,
        irreps_node: str,
        irreps_sh: str,
        rbf_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_sh = o3.Irreps(irreps_sh)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_node,
            shared_weights=False,
        )
        self.weight_mlp = RadialMLP(rbf_dim, hidden_dim, self.tp.weight_numel)
        self.lin_self = o3.Linear(self.irreps_node, self.irreps_node)
        self.lin_out = o3.Linear(self.irreps_node, self.irreps_node)

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> torch.Tensor:
        if edge_src.numel() == 0:
            return x

        weights = self.weight_mlp(edge_rbf)
        m = self.tp(x[edge_src], edge_sh, weights)

        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_dst, m)

        # degree normalization for stability
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        deg.index_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(-1)
        agg = agg / deg

        return self.lin_self(x) + self.lin_out(agg)


class TrueE3EEAtomEncoder(nn.Module):
    def __init__(
        self,
        max_z: int = 100,
        cutoff: float = 6.0,
        num_interactions: int = 3,
        rbf_dim: int = 16,
        lmax: int = 2,
        node_attr_dim: int = 64,
        hidden_dim: int = 128,
        irreps_node: str = "64x0e + 32x1o + 16x2e",
    ):
        super().__init__()
        self.cutoff = cutoff
        self.rbf_dim = rbf_dim
        self.irreps_node = o3.Irreps(irreps_node)

        self.graph_builder = RadiusGraphBuilder(cutoff=cutoff)
        self.dist_rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.z_emb = nn.Embedding(max_z + 1, node_attr_dim)

        self.input_scalar_dim = node_attr_dim + 1 + rbf_dim
        self.input_lin = o3.Linear(
            irreps_in=o3.Irreps(f"{self.input_scalar_dim}x0e"),
            irreps_out=self.irreps_node,
        )

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        self.blocks = nn.ModuleList([
            EquivariantInteractionBlock(
                irreps_node=str(self.irreps_node),
                irreps_sh=str(self.irreps_sh),
                rbf_dim=rbf_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(num_interactions)
        ])

        self.out_norm = nn.LayerNorm(self.irreps_node.dim)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        absorber_index: int = 0,
    ) -> torch.Tensor:
        device = pos.device
        B, N = z.shape

        geom = build_absorber_relative_geometry(z, pos, mask, absorber_index=absorber_index)
        r_abs = geom["r"]

        abs_flag = torch.zeros_like(z, dtype=pos.dtype)
        abs_flag[:, absorber_index] = 1.0

        zf = self.z_emb(z)
        rf = self.dist_rbf(r_abs.clamp(max=self.cutoff))

        scalar_in = torch.cat([zf, abs_flag.unsqueeze(-1), rf], dim=-1)
        scalar_in = scalar_in.reshape(B * N, self.input_scalar_dim)

        x = self.input_lin(scalar_in)

        flat_mask = mask.reshape(B * N)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)

        edge_src, edge_dst, edge_vec = self.graph_builder(pos, mask)

        if edge_vec.numel() > 0:
            edge_len = torch.linalg.norm(edge_vec, dim=-1)
            edge_dir = edge_vec / edge_len.unsqueeze(-1).clamp_min(1e-8)
            edge_rbf = self.dist_rbf(edge_len.clamp(max=self.cutoff))
            edge_sh = o3.spherical_harmonics(
                self.irreps_sh,
                edge_dir,
                normalize=True,
                normalization="component",
            )
        else:
            edge_rbf = torch.zeros(0, self.rbf_dim, device=device, dtype=pos.dtype)
            edge_sh = torch.zeros(0, self.irreps_sh.dim, device=device, dtype=pos.dtype)

        for block in self.blocks:
            x = x + block(
                x=x,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_sh=edge_sh,
                edge_rbf=edge_rbf,
            )

        x = self.out_norm(x)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)

        return x.view(B, N, self.irreps_node.dim)


# ============================================================
# Energy-conditioned scattering heads
# ============================================================

class ElementEnergyScattering(nn.Module):
    def __init__(
        self,
        max_z: int,
        z_emb_dim: int,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.mlp = MLP(
            in_dim=z_emb_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, z_neigh: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        B, M = z_neigh.shape
        nE, dE = e_feat.shape

        zf = self.z_emb(z_neigh).unsqueeze(2).expand(B, M, nE, -1)
        ef = e_feat.unsqueeze(0).unsqueeze(0).expand(B, M, nE, dE)

        return self.mlp(torch.cat([zf, ef], dim=-1))


class PairElementEnergyScattering(nn.Module):
    def __init__(
        self,
        max_z: int,
        z_emb_dim: int,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.mlp = MLP(
            in_dim=2 * z_emb_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, z_j: torch.Tensor, z_k: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        B, P = z_j.shape
        nE, dE = e_feat.shape

        ej = self.z_emb(z_j).unsqueeze(2).expand(B, P, nE, -1)
        ek = self.z_emb(z_k).unsqueeze(2).expand(B, P, nE, -1)
        ef = e_feat.unsqueeze(0).unsqueeze(0).expand(B, P, nE, dE)

        return self.mlp(torch.cat([ej, ek, ef], dim=-1))


class AbsorberNeighbourAggregator(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        rbf_dim: int,
        geom_hidden_dim: int,
        scatter_dim: int,
        out_dim: int,
        cutoff: float,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = GaussianRBF(0.0, cutoff, rbf_dim)

        self.geom_mlp = MLP(
            in_dim=2 * atom_dim + rbf_dim + 3,
            hidden_dim=geom_hidden_dim,
            out_dim=scatter_dim,
            n_layers=3,
        )

        self.out_proj = MLP(
            in_dim=scatter_dim,
            hidden_dim=geom_hidden_dim,
            out_dim=out_dim,
            n_layers=2,
        )

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        elem_energy: ElementEnergyScattering,
        e_feat: torch.Tensor,
        absorber_index: int = 0,
    ) -> torch.Tensor:
        B, N, H = h.shape

        geom = build_absorber_relative_geometry(z, pos, mask, absorber_index)
        r = geom["r"]
        u = geom["u"]
        valid = geom["valid_neigh"] & (r <= self.cutoff)

        h0 = h[:, absorber_index, :].unsqueeze(1).expand(B, N, H)
        hr = self.rbf(r.clamp(max=self.cutoff))

        geom_in = torch.cat([h0, h, hr, u], dim=-1)
        g_geom = self.geom_mlp(geom_in)              # (B,N,D)
        g_elem = elem_energy(z, e_feat)              # (B,N,nE,D)

        contrib = g_geom.unsqueeze(2) * g_elem
        contrib = contrib * valid.unsqueeze(-1).unsqueeze(-1).to(contrib.dtype)

        agg = contrib.sum(dim=1)                     # (B,nE,D)
        return self.out_proj(agg)                    # (B,nE,L)


class AbsorberPathAggregator(nn.Module):
    """
    3-body absorber-centred path aggregator for paths (0, j, k).
    """

    def __init__(
        self,
        atom_dim: int,
        rbf_dim: int,
        geom_hidden_dim: int,
        scatter_dim: int,
        out_dim: int,
        cutoff: float,
        max_paths_per_structure: int = 256,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_paths_per_structure = max_paths_per_structure
        self.rbf = GaussianRBF(0.0, cutoff, rbf_dim)

        self.geom_mlp = MLP(
            in_dim=2 * atom_dim + 3 * rbf_dim + 1,
            hidden_dim=geom_hidden_dim,
            out_dim=scatter_dim,
            n_layers=3,
        )

        self.out_proj = MLP(
            in_dim=scatter_dim,
            hidden_dim=geom_hidden_dim,
            out_dim=out_dim,
            n_layers=2,
        )

    def _enumerate_paths(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        absorber_index: int,
    ):
        B, N = z.shape
        device = z.device

        geom = build_absorber_relative_geometry(z, pos, mask, absorber_index)
        r = geom["r"]
        valid = geom["valid_neigh"] & (r <= self.cutoff)

        all_j = []
        all_k = []
        all_m = []
        Pmax = self.max_paths_per_structure

        for b in range(B):
            idx = torch.where(valid[b])[0]

            if idx.numel() < 2:
                j_idx = torch.zeros(Pmax, dtype=torch.long, device=device)
                k_idx = torch.zeros(Pmax, dtype=torch.long, device=device)
                pmask = torch.zeros(Pmax, dtype=torch.bool, device=device)
            else:
                pairs = torch.combinations(idx, r=2)
                if pairs.shape[0] > Pmax:
                    pairs = pairs[:Pmax]
                P = pairs.shape[0]

                j_idx = torch.zeros(Pmax, dtype=torch.long, device=device)
                k_idx = torch.zeros(Pmax, dtype=torch.long, device=device)
                pmask = torch.zeros(Pmax, dtype=torch.bool, device=device)

                j_idx[:P] = pairs[:, 0]
                k_idx[:P] = pairs[:, 1]
                pmask[:P] = True

            all_j.append(j_idx)
            all_k.append(k_idx)
            all_m.append(pmask)

        return torch.stack(all_j, dim=0), torch.stack(all_k, dim=0), torch.stack(all_m, dim=0)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        pair_elem_energy: PairElementEnergyScattering,
        e_feat: torch.Tensor,
        absorber_index: int = 0,
    ) -> torch.Tensor:
        B, N, H = h.shape
        device = h.device

        j_idx, k_idx, path_mask = self._enumerate_paths(z, pos, mask, absorber_index)
        batch_idx = torch.arange(B, device=device)[:, None]

        hj = h[batch_idx, j_idx]  # (B,P,H)
        hk = h[batch_idx, k_idx]  # (B,P,H)

        pos0 = pos[:, absorber_index, :].unsqueeze(1)
        posj = pos[batch_idx, j_idx]
        posk = pos[batch_idx, k_idx]

        vj = posj - pos0
        vk = posk - pos0
        vjk = posk - posj

        r0j = torch.linalg.norm(vj, dim=-1)
        r0k = torch.linalg.norm(vk, dim=-1)
        rjk = torch.linalg.norm(vjk, dim=-1)

        uj = vj / r0j.unsqueeze(-1).clamp_min(1e-8)
        uk = vk / r0k.unsqueeze(-1).clamp_min(1e-8)
        cosang = (uj * uk).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        f0j = self.rbf(r0j.clamp(max=self.cutoff))
        f0k = self.rbf(r0k.clamp(max=self.cutoff))
        fjk = self.rbf(rjk.clamp(max=self.cutoff))

        geom_in = torch.cat([hj, hk, f0j, f0k, fjk, cosang], dim=-1)
        g_geom = self.geom_mlp(geom_in)                         # (B,P,D)

        zj = z[batch_idx, j_idx]
        zk = z[batch_idx, k_idx]
        g_elem = pair_elem_energy(zj, zk, e_feat)              # (B,P,nE,D)

        contrib = g_geom.unsqueeze(2) * g_elem
        contrib = contrib * path_mask.unsqueeze(-1).unsqueeze(-1).to(contrib.dtype)

        agg = contrib.sum(dim=1)                               # (B,nE,D)
        return self.out_proj(agg)                              # (B,nE,L)


# ============================================================
# Final model
# ============================================================

@register_model("e3eenet")
@register_scheme("e3eenet", scheme_name="e3ee")
class E3EEmbed(Model):
    """
    Absorber-centred energy embedded e3 XANES model with neighbour and optional path scattering.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_z: int = 100,
        atom_emb_dim: int = 64,
        atom_hidden_dim: int = 128,
        atom_layers: int = 3,
        local_cutoff: float = 6.0,
        rbf_dim: int = 32,
        energy_rbf_dim: int = 48,
        scatter_dim: int = 128,
        latent_dim: int = 128,
        head_hidden_dim: int = 128,
        e3nn_irreps: str = "64x0e + 32x1o + 16x2e",
        e3nn_lmax: int = 2,
        out_mlp_layers: int = 3,
        use_path_terms: bool = False,
        max_paths_per_structure: int = 128,
    ):
        super().__init__()
        self.nn_flag = 1
        self.gnn_flag = 0
        self.batch_flag = 1
        self.use_path_terms = use_path_terms

        self.atom_encoder = TrueE3EEAtomEncoder(
            max_z=max_z,
            cutoff=local_cutoff,
            num_interactions=atom_layers,
            rbf_dim=rbf_dim,
            lmax=e3nn_lmax,
            node_attr_dim=atom_emb_dim,
            hidden_dim=atom_hidden_dim,
            irreps_node=e3nn_irreps,
        )

        scalar_idx = scalar_indices_from_irreps(self.atom_encoder.irreps_node)
        self.register_buffer("scalar_idx", torch.tensor(scalar_idx, dtype=torch.long))
        scalar_dim = len(scalar_idx)

        self.energy_embedding = EnergyRBFEmbedding(
            e_min=0.0,
            e_max=max(float(out_features - 1), 1.0),
            n_rbf=energy_rbf_dim,
        )

        self.elem_energy = ElementEnergyScattering(
            max_z=max_z,
            z_emb_dim=32,
            e_dim=energy_rbf_dim,
            hidden_dim=128,
            out_dim=scatter_dim,
        )

        self.neigh_agg = AbsorberNeighbourAggregator(
            atom_dim=scalar_dim,
            rbf_dim=rbf_dim,
            geom_hidden_dim=128,
            scatter_dim=scatter_dim,
            out_dim=latent_dim,
            cutoff=local_cutoff,
        )

        if self.use_path_terms:
            self.pair_elem_energy = PairElementEnergyScattering(
                max_z=max_z,
                z_emb_dim=32,
                e_dim=energy_rbf_dim,
                hidden_dim=128,
                out_dim=scatter_dim,
            )

            self.path_agg = AbsorberPathAggregator(
                atom_dim=scalar_dim,
                rbf_dim=rbf_dim,
                geom_hidden_dim=128,
                scatter_dim=scatter_dim,
                out_dim=latent_dim,
                cutoff=local_cutoff,
                max_paths_per_structure=max_paths_per_structure,
            )

        self.abs_proj = MLP(
            in_dim=scalar_dim,
            hidden_dim=head_hidden_dim,
            out_dim=latent_dim,
            n_layers=2,
        )

        head_in_dim = 3 * latent_dim if self.use_path_terms else 2 * latent_dim
        self.head = MLP(
            in_dim=head_in_dim,
            hidden_dim=head_hidden_dim,
            out_dim=1,
            n_layers=out_mlp_layers,
        )

        self.register_config(
            {
                "in_features": in_features,
                "out_features": out_features,
                "max_z": max_z,
                "atom_emb_dim": atom_emb_dim,
                "atom_hidden_dim": atom_hidden_dim,
                "atom_layers": atom_layers,
                "local_cutoff": local_cutoff,
                "rbf_dim": rbf_dim,
                "energy_rbf_dim": energy_rbf_dim,
                "scatter_dim": scatter_dim,
                "latent_dim": latent_dim,
                "head_hidden_dim": head_hidden_dim,
                "e3nn_irreps": e3nn_irreps,
                "e3nn_lmax": e3nn_lmax,
                "out_mlp_layers": out_mlp_layers,
                "use_path_terms": use_path_terms,
                "max_paths_per_structure": max_paths_per_structure,
            },
            type="e3eenet",
        )

    def forward(self, batch):
        z = batch.z
        pos = batch.pos
        mask = batch.mask
        absorber_index = int(batch.absorber_index.item()) if batch.absorber_index is not None else 0

        if getattr(batch, "e", None) is not None:
            energies = batch.e
            if energies.ndim > 1:
                energies = energies[0]
            energies = energies.to(pos.device)
            energies = torch.linspace(
                0.0,
                float(len(energies) - 1),
                len(energies),
                device=pos.device,
                dtype=pos.dtype,
            )
        else:
            energies = torch.arange(
                batch.y.shape[-1],
                device=pos.device,
                dtype=pos.dtype,
            )

        h_full = self.atom_encoder(z, pos, mask, absorber_index=absorber_index)
        h = torch.index_select(h_full, dim=-1, index=self.scalar_idx)

        e_feat = self.energy_embedding(energies)

        neigh_lat = self.neigh_agg(
            h=h,
            z=z,
            pos=pos,
            mask=mask,
            elem_energy=self.elem_energy,
            e_feat=e_feat,
            absorber_index=absorber_index,
        )  # (B,nE,L)

        abs_lat = self.abs_proj(h[:, absorber_index, :])        # (B,L)
        abs_lat = abs_lat.unsqueeze(1).expand(-1, len(energies), -1)

        parts = [abs_lat, neigh_lat]

        if self.use_path_terms:
            path_lat = self.path_agg(
                h=h,
                z=z,
                pos=pos,
                mask=mask,
                pair_elem_energy=self.pair_elem_energy,
                e_feat=e_feat,
                absorber_index=absorber_index,
            )  # (B,nE,L)
            parts.append(path_lat)

        x = torch.cat(parts, dim=-1)
        y = self.head(x).squeeze(-1)                            # (B,nE)
        return y
