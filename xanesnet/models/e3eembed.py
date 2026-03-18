"""
Absorber-centred e3ee XANES model for XANESNET.

Revised version with:
- smooth cosine cutoff envelope
- irrep-aware normalization
- gentler residual updates
- weighted message aggregation in equivariant encoder
- invariant summaries from all irreps for downstream use
- energy-conditioned attention over atomwise invariant features
- optional path branch retained
"""

from typing import Optional, List

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate
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


def invariant_feature_dim(irreps: o3.Irreps) -> int:
    return sum(mul for mul, _ in irreps)


def invariant_features_from_irreps(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """
    Convert flattened irreps features into invariant features.

    For l = 0:
        keep scalar channels directly

    For l > 0:
        take RMS norm of each irrep copy
    """
    outs = []
    offset = 0
    B, N, _ = x.shape

    for mul, ir in irreps:
        dim = ir.dim
        block_dim = mul * dim
        xb = x[:, :, offset:offset + block_dim].view(B, N, mul, dim)

        if ir.l == 0:
            outs.append(xb.view(B, N, mul))
        else:
            inv = torch.sqrt((xb ** 2).mean(dim=-1) + 1e-8) 
            outs.append(inv)

        offset += block_dim

    return torch.cat(outs, dim=-1)


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


class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r / self.cutoff
        out = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        out = out * (r <= self.cutoff).to(r.dtype)
        return out


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


class IrrepNorm(nn.Module):
    """
    Simple irrep-respecting normalization.

    - l = 0 blocks:
        mean/variance normalization across multiplicity channels
    - l > 0 blocks:
        RMS normalization per irrep copy
    """

    def __init__(self, irreps: o3.Irreps, eps: float = 1e-8, affine: bool = True):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(self.irreps.dim))
            self.bias = nn.Parameter(torch.zeros(self.irreps.dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        offset = 0
        B = x.shape[0]

        for mul, ir in self.irreps:
            dim = ir.dim
            block_dim = mul * dim
            xb = x[:, offset:offset + block_dim].view(B, mul, dim)

            if ir.l == 0:
                mean = xb.mean(dim=1, keepdim=True)
                var = ((xb - mean) ** 2).mean(dim=1, keepdim=True)
                xb = (xb - mean) / torch.sqrt(var + self.eps)
            else:
                norm = torch.sqrt((xb ** 2).mean(dim=2, keepdim=True) + self.eps)
                xb = xb / norm

            outs.append(xb.reshape(B, block_dim))
            offset += block_dim

        out = torch.cat(outs, dim=-1)

        if self.affine:
            out = out * self.weight + self.bias

        return out

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

# ============================================================
# Equivariant encoder
# ============================================================

class EquivariantInteractionBlock(nn.Module):
    def __init__(
        self,
        irreps_node: str,
        irreps_sh: str,
        irreps_message: str,
        rbf_dim: int,
        radial_hidden_dim: int,
        cutoff: float,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_message = o3.Irreps(irreps_message)

        self.pre_norm = IrrepNorm(self.irreps_node)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_message,
            shared_weights=False,
        )
        self.weight_mlp = RadialMLP(rbf_dim, radial_hidden_dim, self.tp.weight_numel)

        self.edge_gate = nn.Sequential(
            nn.Linear(rbf_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, 1),
            nn.Sigmoid(),
        )

        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_message if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_message if ir.l > 0])
        irreps_gates = (
            o3.Irreps(f"{irreps_gated.num_irreps}x0e")
            if irreps_gated.num_irreps > 0
            else o3.Irreps("")
        )

        self.msg_linear = o3.Linear(
            self.irreps_message,
            irreps_scalars + irreps_gates + irreps_gated,
        )

        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[torch.nn.functional.silu] * len(irreps_scalars),
            irreps_gates=irreps_gates,
            act_gates=[torch.sigmoid] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )

        self.update_linear = o3.Linear(self.gate.irreps_out, self.irreps_node)
        self.self_linear = o3.Linear(self.irreps_node, self.irreps_node)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> torch.Tensor:
        if edge_src.numel() == 0:
            return x

        x_norm = self.pre_norm(x)

        tp_weights = self.weight_mlp(edge_rbf)
        m = self.tp(x_norm[edge_src], edge_sh, tp_weights)

        cutoff_w = self.cutoff_fn(edge_len).unsqueeze(-1)
        gate_w = self.edge_gate(edge_rbf)
        edge_w = cutoff_w * gate_w

        m = m * edge_w

        agg = torch.zeros(
            x.shape[0],
            self.irreps_message.dim,
            device=x.device,
            dtype=x.dtype,
        )
        agg.index_add_(0, edge_dst, m)

        norm = torch.zeros(
            x.shape[0],
            1,
            device=x.device,
            dtype=x.dtype,
        )
        norm.index_add_(0, edge_dst, edge_w)
        agg = agg / norm.clamp_min(1e-8)

        agg = self.msg_linear(agg)
        agg = self.gate(agg)

        out = self.self_linear(x_norm) + self.update_linear(agg)
        return x + self.res_scale.to(x.dtype) * out


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
        irreps_message: str = "16x0e + 8x1o + 4x2e",
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.rbf_dim = rbf_dim
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_message = o3.Irreps(irreps_message)

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
                irreps_message=str(self.irreps_message),
                rbf_dim=rbf_dim,
                radial_hidden_dim=hidden_dim,
                cutoff=cutoff,
                residual_scale_init=residual_scale_init,
            )
            for _ in range(num_interactions)
        ])

        self.out_norm = IrrepNorm(self.irreps_node)

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
            edge_len = torch.zeros(0, device=device, dtype=pos.dtype)
            edge_rbf = torch.zeros(0, self.rbf_dim, device=device, dtype=pos.dtype)
            edge_sh = torch.zeros(0, self.irreps_sh.dim, device=device, dtype=pos.dtype)

        for block in self.blocks:
            x = block(
                x=x,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_sh=edge_sh,
                edge_rbf=edge_rbf,
                edge_len=edge_len,
            )

        x = self.out_norm(x)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)

        return x.view(B, N, self.irreps_node.dim)

# ============================================================
# Optional path branch
# ============================================================

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


class AbsorberPathAggregator(nn.Module):
    """
    3-body absorber-centred path aggregator for paths (0, j, k),
    operating on invariant atomwise features.
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
        self.cutoff_fn = CosineCutoff(cutoff)

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

    def _enumerate_paths(self, z, pos, mask, absorber_index):
        B, N = z.shape
        device = z.device

        geom = build_absorber_relative_geometry(z, pos, mask, absorber_index)
        r = geom["r"]
        valid = geom["valid_neigh"] & (r <= self.cutoff)

        all_j, all_k, all_m = [], [], []
        Pmax = self.max_paths_per_structure

        for b in range(B):
            idx = torch.where(valid[b])[0]

            if idx.numel() < 2:
                j_idx = torch.zeros(Pmax, dtype=torch.long, device=device)
                k_idx = torch.zeros(Pmax, dtype=torch.long, device=device)
                pmask = torch.zeros(Pmax, dtype=torch.bool, device=device)
            else:
                pairs = torch.combinations(idx, r=2)

                pos0 = pos[b, absorber_index].unsqueeze(0)
                posj = pos[b, pairs[:, 0]]
                posk = pos[b, pairs[:, 1]]

                r0j = torch.linalg.norm(posj - pos0, dim=-1)
                r0k = torch.linalg.norm(posk - pos0, dim=-1)
                rjk = torch.linalg.norm(posk - posj, dim=-1)

                score = r0j + r0k + 0.5 * rjk
                order = torch.argsort(score)
                pairs = pairs[order]

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

        hj = h[batch_idx, j_idx]
        hk = h[batch_idx, k_idx]

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
        g_geom = self.geom_mlp(geom_in)

        zj = z[batch_idx, j_idx]
        zk = z[batch_idx, k_idx]
        g_elem = pair_elem_energy(zj, zk, e_feat)

        cutoff_w = (
            self.cutoff_fn(r0j) *
            self.cutoff_fn(r0k) *
            self.cutoff_fn(rjk)
        ).unsqueeze(-1).unsqueeze(-1)

        contrib = g_geom.unsqueeze(2) * g_elem
        contrib = contrib * cutoff_w
        contrib = contrib * path_mask.unsqueeze(-1).unsqueeze(-1).to(contrib.dtype)

        agg = contrib.sum(dim=1)

        norm = cutoff_w.squeeze(-1) * path_mask.unsqueeze(-1).to(cutoff_w.dtype)
        norm = norm.sum(dim=1).clamp_min(1e-8)
        agg = agg / norm.unsqueeze(1)

        return self.out_proj(agg)

# ============================================================
# Energy-conditioned atomwise attention
# ============================================================

class EnergyConditionedAtomAttention(nn.Module):
    """
    Energy-conditioned attention over atomwise invariant features.

    For each energy:
    - build an energy-dependent query from absorber invariant features + energy embedding
    - build atom keys/values from atom invariant features + geometry
    - attend over atoms to obtain an energy-dependent latent vector
    """

    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        rbf_dim: int,
        hidden_dim: int,
        latent_dim: int,
        cutoff: float,
        max_z: int = 100,
        z_emb_dim: int = 32,
        n_heads: int = 4,
    ):
        super().__init__()
        assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"

        self.atom_dim = atom_dim
        self.e_dim = e_dim
        self.rbf_dim = rbf_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cutoff = cutoff
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads

        self.rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)

        self.query_mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        self.key_mlp = MLP(
            in_dim=atom_dim + z_emb_dim + rbf_dim + 3 + 1 + e_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        self.value_mlp = MLP(
            in_dim=atom_dim + z_emb_dim + rbf_dim + 3 + 1 + e_dim,
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

        self.score_scale = self.head_dim ** -0.5

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.shape[:-1] + (self.n_heads, self.head_dim)
        return x.view(*new_shape)

    def forward(
        self,
        h: torch.Tensor,          
        z: torch.Tensor,          
        pos: torch.Tensor,        
        mask: torch.Tensor,       
        e_feat: torch.Tensor,     
        absorber_index: int = 0,
    ) -> torch.Tensor:
        B, N, H = h.shape
        nE, dE = e_feat.shape
        device = h.device
        dtype = h.dtype

        geom = build_absorber_relative_geometry(z, pos, mask, absorber_index)
        r = geom["r"]                              
        u = geom["u"]                              
        valid = mask & (r <= self.cutoff)          

        h_abs = h[:, absorber_index, :]            
        q_in = torch.cat([
            h_abs.unsqueeze(1).expand(B, nE, H),
            e_feat.unsqueeze(0).expand(B, nE, dE),
        ], dim=-1)                                 

        q = self.query_mlp(q_in)                   
        q = self._split_heads(q)                   

        zr = self.z_emb(z)                         
        rr = self.rbf(r.clamp(max=self.cutoff))    
        cutoff_w = self.cutoff_fn(r)               

        is_abs = torch.zeros_like(r)
        is_abs[:, absorber_index] = 1.0

        atom_base = torch.cat([h, zr, rr, u, is_abs.unsqueeze(-1)], dim=-1)  
        atom_base = atom_base.unsqueeze(2).expand(B, N, nE, atom_base.shape[-1])
        ef = e_feat.unsqueeze(0).unsqueeze(0).expand(B, N, nE, dE)

        kv_in = torch.cat([atom_base, ef], dim=-1)  
        k = self.key_mlp(kv_in)                     
        v = self.value_mlp(kv_in)                   

        k = k.permute(0, 2, 1, 3).contiguous()      
        v = v.permute(0, 2, 1, 3).contiguous()      

        k = self._split_heads(k)                    
        v = self._split_heads(v)                    

        scores = (q.unsqueeze(2) * k).sum(dim=-1) * self.score_scale

        radial_bias = torch.log(cutoff_w.clamp_min(1e-8)).unsqueeze(1).unsqueeze(-1)  
        scores = scores + radial_bias

        attn_mask = valid.unsqueeze(1).unsqueeze(-1)  
        scores = scores.masked_fill(~attn_mask, -1e9)

        attn = torch.softmax(scores, dim=2)        
        attn = attn * attn_mask.to(attn.dtype)
        out = (attn.unsqueeze(-1) * v).sum(dim=2)   
        out = out.reshape(B, nE, self.latent_dim)   

        return self.out_proj(out)


class EnergyConditionedAbsorberBranch(nn.Module):
    """
    Energy-dependent absorber branch.
    """

    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, h_abs: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        B, H = h_abs.shape
        nE, dE = e_feat.shape

        ha = h_abs.unsqueeze(1).expand(B, nE, H)
        ef = e_feat.unsqueeze(0).expand(B, nE, dE)
        return self.mlp(torch.cat([ha, ef], dim=-1))


# ============================================================
# Final model
# ============================================================

@register_model("e3eenet")
@register_scheme("e3eenet", scheme_name="e3ee")
class E3EEmbed(Model):
    """
    Absorber-centred energy embedded e3 XANES model with:
    - equivariant atom encoder
    - invariant atomwise summaries
    - energy-conditioned attention over atoms
    - optional path branch
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
        e3nn_irreps_message: str = "16x0e + 8x1o + 4x2e",
        e3nn_lmax: int = 2,
        out_mlp_layers: int = 3,
        use_path_terms: bool = False,
        max_paths_per_structure: int = 128,
        residual_scale_init: float = 0.1,
        attention_heads: int = 4,
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
            irreps_message=e3nn_irreps_message,
            residual_scale_init=residual_scale_init,
        )

        self.inv_dim = invariant_feature_dim(self.atom_encoder.irreps_node)

        self.energy_embedding = EnergyRBFEmbedding(
            e_min=0.0,
            e_max=max(float(out_features - 1), 1.0),
            n_rbf=energy_rbf_dim,
        )

        self.abs_branch = EnergyConditionedAbsorberBranch(
            atom_dim=self.inv_dim,
            e_dim=energy_rbf_dim,
            hidden_dim=head_hidden_dim,
            out_dim=latent_dim,
        )

        self.atom_attention = EnergyConditionedAtomAttention(
            atom_dim=self.inv_dim,
            e_dim=energy_rbf_dim,
            rbf_dim=rbf_dim,
            hidden_dim=atom_hidden_dim,
            latent_dim=latent_dim,
            cutoff=local_cutoff,
            max_z=max_z,
            z_emb_dim=32,
            n_heads=attention_heads,
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
                atom_dim=self.inv_dim,
                rbf_dim=rbf_dim,
                geom_hidden_dim=128,
                scatter_dim=scatter_dim,
                out_dim=latent_dim,
                cutoff=local_cutoff,
                max_paths_per_structure=max_paths_per_structure,
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
                "e3nn_irreps_message": e3nn_irreps_message,
                "e3nn_lmax": e3nn_lmax,
                "out_mlp_layers": out_mlp_layers,
                "use_path_terms": use_path_terms,
                "max_paths_per_structure": max_paths_per_structure,
                "residual_scale_init": residual_scale_init,
                "attention_heads": attention_heads,
            },
            type="e3eenet",
        )

    def forward(self, batch):
        z = batch.z
        pos = batch.pos
        mask = batch.mask

        absorber_index = 0

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
        h = invariant_features_from_irreps(h_full, self.atom_encoder.irreps_node)  

        e_feat = self.energy_embedding(energies)  

        abs_lat = self.abs_branch(h[:, absorber_index, :], e_feat)  
        attn_lat = self.atom_attention(
            h=h,
            z=z,
            pos=pos,
            mask=mask,
            e_feat=e_feat,
            absorber_index=absorber_index,
        )  

        parts = [abs_lat, attn_lat]

        if self.use_path_terms:
            path_lat = self.path_agg(
                h=h,
                z=z,
                pos=pos,
                mask=mask,
                pair_elem_energy=self.pair_elem_energy,
                e_feat=e_feat,
                absorber_index=absorber_index,
            )
            parts.append(path_lat)

        x = torch.cat(parts, dim=-1)
        y = self.head(x).squeeze(-1)
        return y
