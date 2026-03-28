from typing import Optional

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
    orig_shape = x.shape[:-1]
    D = x.shape[-1]

    x = x.reshape(-1, D)
    outs = []
    offset = 0
    M = x.shape[0]

    for mul, ir in irreps:
        dim = ir.dim
        block_dim = mul * dim
        xb = x[:, offset:offset + block_dim].reshape(M, mul, dim)

        if ir.l == 0:
            outs.append(xb.reshape(M, mul))
        else:
            inv = torch.sqrt((xb ** 2).mean(dim=-1) + 1e-8)
            outs.append(inv)

        offset += block_dim

    out = torch.cat(outs, dim=-1)
    return out.view(*orig_shape, out.shape[-1])


class GaussianRBF(nn.Module):
    def __init__(self, start: float, stop: float, n_rbf: int, gamma: Optional[float] = None):
        super().__init__()
        centers = torch.linspace(start, stop, n_rbf)
        self.register_buffer("centers", centers)
        if gamma is None:
            delta = (stop - start) / max(n_rbf - 1, 1)
            gamma = 1.0 / (delta * delta + 1e-12)
        self.gamma = float(gamma)

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
        orig_shape = x.shape[:-1]
        D = x.shape[-1]
        x = x.reshape(-1, D)

        outs = []
        offset = 0
        Bflat = x.shape[0]

        for mul, ir in self.irreps:
            dim = ir.dim
            block_dim = mul * dim
            xb = x[:, offset:offset + block_dim].reshape(Bflat, mul, dim)

            if ir.l == 0:
                mean = xb.mean(dim=1, keepdim=True)
                var = ((xb - mean) ** 2).mean(dim=1, keepdim=True)
                xb = (xb - mean) / torch.sqrt(var + self.eps)
            else:
                norm = torch.sqrt((xb ** 2).mean(dim=2, keepdim=True) + self.eps)
                xb = xb / norm

            outs.append(xb.reshape(Bflat, block_dim))
            offset += block_dim

        out = torch.cat(outs, dim=-1)

        if self.affine:
            out = out * self.weight + self.bias

        return out.view(*orig_shape, D)


# ============================================================
# Batched/vectorized graph builder
# ============================================================


class BatchedRadiusGraphBuilder(nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, pos: torch.Tensor, mask: torch.Tensor):
        device = pos.device
        dtype = pos.dtype
        B, N, _ = pos.shape

        diff = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)

        valid = mask[:, :, None] & mask[:, None, :]
        edge_mask = valid & (dist <= self.cutoff) & (dist > 1e-8)

        b, src, dst = torch.where(edge_mask)

        if b.numel() == 0:
            edge_src = torch.zeros(0, dtype=torch.long, device=device)
            edge_dst = torch.zeros(0, dtype=torch.long, device=device)
            edge_vec = torch.zeros(0, 3, dtype=dtype, device=device)
        else:
            edge_src = b * N + src
            edge_dst = b * N + dst
            edge_vec = pos[b, dst] - pos[b, src]

        return edge_src, edge_dst, edge_vec


# ============================================================
# Invariant encoder
# ============================================================


class InvariantInteractionBlock(nn.Module):
    def __init__(
        self,
        node_dim: int,
        rbf_dim: int,
        radial_hidden_dim: int,
        cutoff: float,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()

        self.pre_norm = nn.LayerNorm(node_dim)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.weight_mlp = RadialMLP(rbf_dim, radial_hidden_dim, node_dim)

        self.edge_gate = nn.Sequential(
            nn.Linear(rbf_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, node_dim),
        )

        self.self_linear = nn.Linear(node_dim, node_dim)
        self.update_linear = nn.Linear(node_dim, node_dim)
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

        edge_w = self.cutoff_fn(edge_len).unsqueeze(-1) * self.edge_gate(edge_rbf)

        m = x_norm[edge_src] * self.weight_mlp(edge_rbf)
        m = m * edge_w

        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_dst, m)

        norm = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        norm.index_add_(0, edge_dst, edge_w)
        agg = agg / norm.clamp_min(1e-8)

        agg = self.msg_mlp(agg)

        out = self.self_linear(x_norm) + self.update_linear(agg)
        return x + self.res_scale.to(x.dtype) * out


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
        use_invariant_block: bool = False,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.rbf_dim = rbf_dim
        self.use_invariant_block = use_invariant_block
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_message = o3.Irreps(irreps_message)

        self.graph_builder = BatchedRadiusGraphBuilder(cutoff=cutoff)
        self.dist_rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.z_emb = nn.Embedding(max_z + 1, node_attr_dim)

        self.input_scalar_dim = node_attr_dim + 1 + rbf_dim
        self.input_lin = o3.Linear(
            irreps_in=o3.Irreps(f"{self.input_scalar_dim}x0e"),
            irreps_out=self.irreps_node,
        )

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        if self.use_invariant_block:
            self.blocks = nn.ModuleList([
                InvariantInteractionBlock(
                    node_dim=self.irreps_node.dim,
                    rbf_dim=rbf_dim,
                    radial_hidden_dim=hidden_dim,
                    cutoff=cutoff,
                    residual_scale_init=residual_scale_init,
                )
                for _ in range(num_interactions)
            ])
            self.out_norm = nn.LayerNorm(self.irreps_node.dim)
        else:
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
        geom: Optional[dict] = None,
    ) -> torch.Tensor:
        device = pos.device
        B, N = z.shape

        if geom is None:
            geom = build_absorber_relative_geometry(
                z=z, pos=pos, mask=mask, absorber_index=absorber_index
            )

        r_abs = geom["r"]

        abs_flag = torch.zeros_like(z, dtype=pos.dtype)
        abs_flag[:, absorber_index] = 1.0

        zf = self.z_emb(z)
        rf = self.dist_rbf(r_abs.clamp(max=self.cutoff))

        scalar_in = torch.cat([zf, abs_flag.unsqueeze(-1), rf], dim=-1)
        x = self.input_lin(scalar_in.reshape(B * N, self.input_scalar_dim))

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
# Absorber branches
# ============================================================


class FieldConditionedAbsorberBranch(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        field_abs_dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.mlp = MLP(
            in_dim=atom_dim + e_dim + field_abs_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, h_abs: torch.Tensor, e_feat: torch.Tensor, field_abs_feat: torch.Tensor) -> torch.Tensor:
        B, H = h_abs.shape
        nE, dE = e_feat.shape
        hf = field_abs_feat.shape[-1]

        ha = h_abs.unsqueeze(1).expand(B, nE, H)
        ef = e_feat.unsqueeze(0).expand(B, nE, dE)
        ff = field_abs_feat.unsqueeze(1).expand(B, nE, hf)

        return self.mlp(torch.cat([ha, ef, ff], dim=-1))


class EnergyConditionedAbsorberBranch(nn.Module):
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


class EnergyIrrepModulation(nn.Module):
    def __init__(self, irreps: o3.Irreps, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.n_copies = sum(mul for mul, _ in self.irreps)

        self.mlp = MLP(
            in_dim=cond_dim,
            hidden_dim=hidden_dim,
            out_dim=self.n_copies,
            n_layers=3,
        )

    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        nE = cond_feat.shape[0] if cond_feat.ndim == 2 else cond_feat.shape[1]

        gates = self.mlp(cond_feat)

        outs = []
        xoff = 0
        goff = 0

        for mul, ir in self.irreps:
            dim = ir.dim
            block_dim = mul * dim

            xb = x[:, xoff:xoff + block_dim].view(B, mul, dim)
            gb = gates[:, goff:goff + mul] if gates.ndim == 2 else gates[..., goff:goff + mul]

            if gb.ndim == 2:
                xb = xb.unsqueeze(1)
                gb = gb.unsqueeze(0).unsqueeze(-1)
            else:
                xb = xb.unsqueeze(1)
                gb = gb.unsqueeze(-1)

            outs.append((xb * gb).reshape(B, nE, block_dim))

            xoff += block_dim
            goff += mul

        return torch.cat(outs, dim=-1)


class FieldConditionedEquivariantAbsorberHead(nn.Module):
    def __init__(
        self,
        irreps_node: o3.Irreps,
        e_dim: int,
        field_abs_dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.inv_dim = invariant_feature_dim(self.irreps_node)

        self.mod = EnergyIrrepModulation(
            self.irreps_node,
            cond_dim=e_dim + field_abs_dim,
            hidden_dim=hidden_dim,
        )

        self.out_mlp = MLP(
            in_dim=self.inv_dim + field_abs_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, h_abs_full: torch.Tensor, e_feat: torch.Tensor, field_abs_feat: torch.Tensor) -> torch.Tensor:
        B = h_abs_full.shape[0]
        nE = e_feat.shape[0]
        ff = field_abs_feat.shape[-1]

        cond = torch.cat([
            e_feat.unsqueeze(0).expand(B, nE, -1),
            field_abs_feat.unsqueeze(1).expand(B, nE, ff),
        ], dim=-1)

        h_mod = self.mod(h_abs_full, cond)
        inv = invariant_features_from_irreps(h_mod, self.irreps_node)
        inv = torch.cat([inv, field_abs_feat.unsqueeze(1).expand(B, nE, ff)], dim=-1)
        return self.out_mlp(inv)


class EnergyConditionedEquivariantAbsorberHead(nn.Module):
    def __init__(
        self,
        irreps_node: o3.Irreps,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.inv_dim = invariant_feature_dim(self.irreps_node)

        self.mod = EnergyIrrepModulation(
            self.irreps_node,
            cond_dim=e_dim,
            hidden_dim=hidden_dim,
        )

        self.out_mlp = MLP(
            in_dim=self.inv_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, h_abs_full: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        B = h_abs_full.shape[0]
        nE = e_feat.shape[0]

        cond = e_feat.unsqueeze(0).expand(B, nE, -1)
        h_mod = self.mod(h_abs_full, cond)
        inv = invariant_features_from_irreps(h_mod, self.irreps_node)
        return self.out_mlp(inv)


# ============================================================
# Invariant attention heads
# ============================================================


class FieldConditionedAtomAttention(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        rbf_dim: int,
        hidden_dim: int,
        latent_dim: int,
        cutoff: float,
        field_atom_dim: int,
        field_abs_dim: int,
        max_z: int = 100,
        z_emb_dim: int = 32,
        n_heads: int = 4,
    ):
        super().__init__()
        assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"

        self.cutoff = cutoff
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads

        self.rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)

        self.query_mlp = MLP(
            in_dim=atom_dim + e_dim + field_abs_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

        atom_static_dim = atom_dim + z_emb_dim + rbf_dim + 3 + 1 + field_atom_dim
        self.key_mlp = MLP(
            in_dim=atom_static_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )
        self.value_mlp = MLP(
            in_dim=atom_static_dim,
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
        return x.view(*x.shape[:-1], self.n_heads, self.head_dim)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        field_atom_lat: torch.Tensor,
        field_abs_feat: torch.Tensor,
        absorber_index: int = 0,
        geom: Optional[dict] = None,
    ) -> torch.Tensor:
        B, N, H = h.shape
        nE, dE = e_feat.shape

        if geom is None:
            geom = build_absorber_relative_geometry(z, pos, mask, absorber_index)

        r = geom["r"]
        u = geom["u"]
        valid = mask & (r <= self.cutoff)

        h_abs = h[:, absorber_index, :]
        faf = field_abs_feat

        q_in = torch.cat([
            h_abs.unsqueeze(1).expand(B, nE, H),
            e_feat.unsqueeze(0).expand(B, nE, dE),
            faf.unsqueeze(1).expand(B, nE, faf.shape[-1]),
        ], dim=-1)
        q = self.query_mlp(q_in)

        zr = self.z_emb(z)
        rr = self.rbf(r.clamp(max=self.cutoff))
        is_abs = torch.zeros_like(r)
        is_abs[:, absorber_index] = 1.0

        atom_static = torch.cat(
            [h, zr, rr, u, is_abs.unsqueeze(-1), field_atom_lat],
            dim=-1,
        )
        k = self.key_mlp(atom_static)
        v = self.value_mlp(atom_static)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1) * self.score_scale
        radial_bias = torch.log(self.cutoff_fn(r).clamp_min(1e-8)).unsqueeze(1).unsqueeze(-1)
        scores = scores + radial_bias

        attn_mask = valid.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~attn_mask, -1e9)

        attn = torch.softmax(scores, dim=2)
        attn = attn * attn_mask.to(attn.dtype)

        out = (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=2)
        out = out.reshape(B, nE, self.latent_dim)

        return self.out_proj(out)


class EnergyConditionedAtomAttention(nn.Module):
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

        self.cutoff = cutoff
        self.latent_dim = latent_dim
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

        atom_static_dim = atom_dim + z_emb_dim + rbf_dim + 3 + 1
        self.key_mlp = MLP(
            in_dim=atom_static_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )
        self.value_mlp = MLP(
            in_dim=atom_static_dim,
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
        return x.view(*x.shape[:-1], self.n_heads, self.head_dim)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        absorber_index: int = 0,
        geom: Optional[dict] = None,
    ) -> torch.Tensor:
        B, N, H = h.shape
        nE, dE = e_feat.shape

        if geom is None:
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

        zr = self.z_emb(z)
        rr = self.rbf(r.clamp(max=self.cutoff))
        is_abs = torch.zeros_like(r)
        is_abs[:, absorber_index] = 1.0

        atom_static = torch.cat([h, zr, rr, u, is_abs.unsqueeze(-1)], dim=-1)
        k = self.key_mlp(atom_static)
        v = self.value_mlp(atom_static)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1) * self.score_scale
        radial_bias = torch.log(self.cutoff_fn(r).clamp_min(1e-8)).unsqueeze(1).unsqueeze(-1)
        scores = scores + radial_bias

        attn_mask = valid.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~attn_mask, -1e9)

        attn = torch.softmax(scores, dim=2)
        attn = attn * attn_mask.to(attn.dtype)

        out = (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=2)
        out = out.reshape(B, nE, self.latent_dim)

        return self.out_proj(out)


# ============================================================
# Equivariant attention heads
# mode 1: energy-independent TP values
# mode 2: low-rank energy-dependent TP weights
# ============================================================


class FieldConditionedEquivariantAtomAttention(nn.Module):
    """
    Equivariant environment branch with energy-independent tensor-product values
    and energy-dependent scalar gating.
    """
    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_sh: o3.Irreps,
        e_dim: int,
        rbf_dim: int,
        hidden_dim: int,
        latent_dim: int,
        cutoff: float,
        field_atom_dim: int,
        field_abs_dim: int,
        max_z: int = 100,
        z_emb_dim: int = 32,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.cutoff = float(cutoff)

        self.inv_dim = invariant_feature_dim(self.irreps_node)

        self.rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)

        self.value_tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_node,
            shared_weights=False,
        )
        self.value_weight_mlp = MLP(
            in_dim=z_emb_dim + rbf_dim + field_atom_dim + field_abs_dim,
            hidden_dim=hidden_dim,
            out_dim=self.value_tp.weight_numel,
            n_layers=3,
        )

        self.score_mlp = MLP(
            in_dim=(
                2 * self.inv_dim
                + z_emb_dim
                + rbf_dim
                + e_dim
                + field_atom_dim
                + field_abs_dim
            ),
            hidden_dim=hidden_dim,
            out_dim=1,
            n_layers=3,
        )

        self.out_mlp = MLP(
            in_dim=self.inv_dim + field_abs_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

    def forward(
        self,
        h_full: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        field_atom_lat: torch.Tensor,
        field_abs_feat: torch.Tensor,
        absorber_index: int = 0,
        geom: Optional[dict] = None,
    ) -> torch.Tensor:
        B, N, D = h_full.shape
        nE, dE = e_feat.shape
        dtype = h_full.dtype

        if geom is None:
            geom = build_absorber_relative_geometry(z, pos, mask, absorber_index)

        r = geom["r"]
        u = geom["u"]
        valid = geom["valid_neigh"] & (r <= self.cutoff)

        inv_abs = invariant_features_from_irreps(h_full[:, absorber_index, :], self.irreps_node)
        inv_nei = invariant_features_from_irreps(h_full, self.irreps_node)

        zr = self.z_emb(z)
        rr = self.rbf(r.clamp(max=self.cutoff))

        sh = o3.spherical_harmonics(
            self.irreps_sh,
            u.reshape(-1, 3),
            normalize=True,
            normalization="component",
        ).view(B, N, self.irreps_sh.dim)

        value_in = torch.cat(
            [
                zr,
                rr,
                field_atom_lat,
                field_abs_feat.unsqueeze(1).expand(B, N, field_abs_feat.shape[-1]),
            ],
            dim=-1,
        )
        tp_w = self.value_weight_mlp(value_in.reshape(B * N, -1))

        values = self.value_tp(
            h_full.reshape(B * N, D),
            sh.reshape(B * N, self.irreps_sh.dim),
            tp_w,
        ).view(B, N, self.irreps_node.dim)

        score_in = torch.cat(
            [
                inv_abs.unsqueeze(1).unsqueeze(1).expand(B, nE, N, self.inv_dim),
                inv_nei.unsqueeze(1).expand(B, nE, N, self.inv_dim),
                zr.unsqueeze(1).expand(B, nE, N, zr.shape[-1]),
                rr.unsqueeze(1).expand(B, nE, N, rr.shape[-1]),
                e_feat.unsqueeze(0).unsqueeze(2).expand(B, nE, N, dE),
                field_atom_lat.unsqueeze(1).expand(B, nE, N, field_atom_lat.shape[-1]),
                field_abs_feat.unsqueeze(1).unsqueeze(2).expand(B, nE, N, field_abs_feat.shape[-1]),
            ],
            dim=-1,
        )

        gate = torch.sigmoid(self.score_mlp(score_in).squeeze(-1))
        gate = gate * self.cutoff_fn(r).unsqueeze(1)
        gate = gate * valid.unsqueeze(1).to(dtype)

        agg = (gate.unsqueeze(-1) * values.unsqueeze(1)).sum(dim=2)
        norm = gate.sum(dim=2, keepdim=True).clamp_min(1e-8)
        agg = agg / norm

        inv_agg = invariant_features_from_irreps(agg, self.irreps_node)
        out_in = torch.cat(
            [
                inv_agg,
                field_abs_feat.unsqueeze(1).expand(B, nE, field_abs_feat.shape[-1]),
            ],
            dim=-1,
        )
        return self.out_mlp(out_in)


class FieldConditionedEquivariantAtomAttentionLowRank(nn.Module):
    """
    Equivariant environment branch with low-rank energy-dependent tensor-product weights:
        w_j(E) = w_base_j + sum_r a_jr * b_r(E) * u_r
    where rank << weight_numel.
    """
    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_sh: o3.Irreps,
        e_dim: int,
        rbf_dim: int,
        hidden_dim: int,
        latent_dim: int,
        cutoff: float,
        field_atom_dim: int,
        field_abs_dim: int,
        tp_rank: int = 8,
        max_z: int = 100,
        z_emb_dim: int = 32,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.cutoff = float(cutoff)
        self.tp_rank = int(tp_rank)

        self.inv_dim = invariant_feature_dim(self.irreps_node)

        self.rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)

        self.value_tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_node,
            shared_weights=False,
        )
        self.weight_numel = self.value_tp.weight_numel

        geom_in_dim = z_emb_dim + rbf_dim + field_atom_dim + field_abs_dim

        self.base_weight_mlp = MLP(
            in_dim=geom_in_dim,
            hidden_dim=hidden_dim,
            out_dim=self.weight_numel,
            n_layers=3,
        )
        self.geom_coeff_mlp = MLP(
            in_dim=geom_in_dim,
            hidden_dim=hidden_dim,
            out_dim=self.tp_rank,
            n_layers=3,
        )
        self.energy_coeff_mlp = MLP(
            in_dim=e_dim + field_abs_dim,
            hidden_dim=hidden_dim,
            out_dim=self.tp_rank,
            n_layers=3,
        )

        self.rank_basis = nn.Parameter(
            torch.randn(self.tp_rank, self.weight_numel) * (self.weight_numel ** -0.5)
        )

        self.score_mlp = MLP(
            in_dim=(
                2 * self.inv_dim
                + z_emb_dim
                + rbf_dim
                + e_dim
                + field_atom_dim
                + field_abs_dim
            ),
            hidden_dim=hidden_dim,
            out_dim=1,
            n_layers=3,
        )

        self.out_mlp = MLP(
            in_dim=self.inv_dim + field_abs_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

    def forward(
        self,
        h_full: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        field_atom_lat: torch.Tensor,
        field_abs_feat: torch.Tensor,
        absorber_index: int = 0,
        geom: Optional[dict] = None,
    ) -> torch.Tensor:
        B, N, D = h_full.shape
        nE, dE = e_feat.shape
        dtype = h_full.dtype

        if geom is None:
            geom = build_absorber_relative_geometry(z, pos, mask, absorber_index)

        r = geom["r"]
        u = geom["u"]
        valid = geom["valid_neigh"] & (r <= self.cutoff)

        inv_abs = invariant_features_from_irreps(h_full[:, absorber_index, :], self.irreps_node)
        inv_nei = invariant_features_from_irreps(h_full, self.irreps_node)

        zr = self.z_emb(z)
        rr = self.rbf(r.clamp(max=self.cutoff))

        sh = o3.spherical_harmonics(
            self.irreps_sh,
            u.reshape(-1, 3),
            normalize=True,
            normalization="component",
        ).view(B, N, self.irreps_sh.dim)

        geom_in = torch.cat(
            [
                zr,
                rr,
                field_atom_lat,
                field_abs_feat.unsqueeze(1).expand(B, N, field_abs_feat.shape[-1]),
            ],
            dim=-1,
        )  # [B,N,G]

        base_w = self.base_weight_mlp(geom_in.reshape(B * N, -1)).view(B, N, self.weight_numel)
        geom_coeff = self.geom_coeff_mlp(geom_in.reshape(B * N, -1)).view(B, N, self.tp_rank)

        energy_in = torch.cat(
            [
                e_feat.unsqueeze(0).expand(B, nE, dE),
                field_abs_feat.unsqueeze(1).expand(B, nE, field_abs_feat.shape[-1]),
            ],
            dim=-1,
        )  # [B,nE,dE+field]
        energy_coeff = self.energy_coeff_mlp(energy_in.reshape(B * nE, -1)).view(B, nE, self.tp_rank)

        # [B,nE,N,R]
        coeff = geom_coeff.unsqueeze(1) * energy_coeff.unsqueeze(2)
        # [B,nE,N,W]
        delta_w = torch.einsum("benr,rw->benw", coeff, self.rank_basis)
        tp_w = base_w.unsqueeze(1) + delta_w

        h_rep = h_full.unsqueeze(1).expand(B, nE, N, D).reshape(B * nE * N, D)
        sh_rep = sh.unsqueeze(1).expand(B, nE, N, self.irreps_sh.dim).reshape(B * nE * N, self.irreps_sh.dim)
        tp_w_rep = tp_w.reshape(B * nE * N, self.weight_numel)

        values = self.value_tp(h_rep, sh_rep, tp_w_rep).view(B, nE, N, self.irreps_node.dim)

        score_in = torch.cat(
            [
                inv_abs.unsqueeze(1).unsqueeze(1).expand(B, nE, N, self.inv_dim),
                inv_nei.unsqueeze(1).expand(B, nE, N, self.inv_dim),
                zr.unsqueeze(1).expand(B, nE, N, zr.shape[-1]),
                rr.unsqueeze(1).expand(B, nE, N, rr.shape[-1]),
                e_feat.unsqueeze(0).unsqueeze(2).expand(B, nE, N, dE),
                field_atom_lat.unsqueeze(1).expand(B, nE, N, field_atom_lat.shape[-1]),
                field_abs_feat.unsqueeze(1).unsqueeze(2).expand(B, nE, N, field_abs_feat.shape[-1]),
            ],
            dim=-1,
        )

        gate = torch.sigmoid(self.score_mlp(score_in).squeeze(-1))
        gate = gate * self.cutoff_fn(r).unsqueeze(1)
        gate = gate * valid.unsqueeze(1).to(dtype)

        agg = (gate.unsqueeze(-1) * values).sum(dim=2)
        norm = gate.sum(dim=2, keepdim=True).clamp_min(1e-8)
        agg = agg / norm

        inv_agg = invariant_features_from_irreps(agg, self.irreps_node)
        out_in = torch.cat(
            [
                inv_agg,
                field_abs_feat.unsqueeze(1).expand(B, nE, field_abs_feat.shape[-1]),
            ],
            dim=-1,
        )
        return self.out_mlp(out_in)


class EnergyConditionedEquivariantAtomAttention(nn.Module):
    """
    Non-field equivariant attention with energy-independent TP values.
    """
    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_sh: o3.Irreps,
        e_dim: int,
        rbf_dim: int,
        hidden_dim: int,
        latent_dim: int,
        cutoff: float,
        max_z: int = 100,
        z_emb_dim: int = 32,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.cutoff = float(cutoff)

        self.inv_dim = invariant_feature_dim(self.irreps_node)

        self.rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)

        self.value_tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_node,
            shared_weights=False,
        )
        self.value_weight_mlp = MLP(
            in_dim=z_emb_dim + rbf_dim,
            hidden_dim=hidden_dim,
            out_dim=self.value_tp.weight_numel,
            n_layers=3,
        )

        self.score_mlp = MLP(
            in_dim=2 * self.inv_dim + z_emb_dim + rbf_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            n_layers=3,
        )

        self.out_mlp = MLP(
            in_dim=self.inv_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

    def forward(
        self,
        h_full: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        absorber_index: int = 0,
        geom: Optional[dict] = None,
    ) -> torch.Tensor:
        B, N, D = h_full.shape
        nE, dE = e_feat.shape
        dtype = h_full.dtype

        if geom is None:
            geom = build_absorber_relative_geometry(z, pos, mask, absorber_index)

        r = geom["r"]
        u = geom["u"]
        valid = geom["valid_neigh"] & (r <= self.cutoff)

        inv_abs = invariant_features_from_irreps(h_full[:, absorber_index, :], self.irreps_node)
        inv_nei = invariant_features_from_irreps(h_full, self.irreps_node)

        zr = self.z_emb(z)
        rr = self.rbf(r.clamp(max=self.cutoff))

        sh = o3.spherical_harmonics(
            self.irreps_sh,
            u.reshape(-1, 3),
            normalize=True,
            normalization="component",
        ).view(B, N, self.irreps_sh.dim)

        value_in = torch.cat([zr, rr], dim=-1)
        tp_w = self.value_weight_mlp(value_in.reshape(B * N, -1))

        values = self.value_tp(
            h_full.reshape(B * N, D),
            sh.reshape(B * N, self.irreps_sh.dim),
            tp_w,
        ).view(B, N, self.irreps_node.dim)

        score_in = torch.cat(
            [
                inv_abs.unsqueeze(1).unsqueeze(1).expand(B, nE, N, self.inv_dim),
                inv_nei.unsqueeze(1).expand(B, nE, N, self.inv_dim),
                zr.unsqueeze(1).expand(B, nE, N, zr.shape[-1]),
                rr.unsqueeze(1).expand(B, nE, N, rr.shape[-1]),
                e_feat.unsqueeze(0).unsqueeze(2).expand(B, nE, N, dE),
            ],
            dim=-1,
        )

        gate = torch.sigmoid(self.score_mlp(score_in).squeeze(-1))
        gate = gate * self.cutoff_fn(r).unsqueeze(1)
        gate = gate * valid.unsqueeze(1).to(dtype)

        agg = (gate.unsqueeze(-1) * values.unsqueeze(1)).sum(dim=2)
        norm = gate.sum(dim=2, keepdim=True).clamp_min(1e-8)
        agg = agg / norm

        inv_agg = invariant_features_from_irreps(agg, self.irreps_node)
        return self.out_mlp(inv_agg)


class EnergyConditionedEquivariantAtomAttentionLowRank(nn.Module):
    """
    Non-field equivariant attention with low-rank energy-dependent TP weights.
    """
    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_sh: o3.Irreps,
        e_dim: int,
        rbf_dim: int,
        hidden_dim: int,
        latent_dim: int,
        cutoff: float,
        tp_rank: int = 8,
        max_z: int = 100,
        z_emb_dim: int = 32,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.cutoff = float(cutoff)
        self.tp_rank = int(tp_rank)

        self.inv_dim = invariant_feature_dim(self.irreps_node)

        self.rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)

        self.value_tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_node,
            shared_weights=False,
        )
        self.weight_numel = self.value_tp.weight_numel

        geom_in_dim = z_emb_dim + rbf_dim

        self.base_weight_mlp = MLP(
            in_dim=geom_in_dim,
            hidden_dim=hidden_dim,
            out_dim=self.weight_numel,
            n_layers=3,
        )
        self.geom_coeff_mlp = MLP(
            in_dim=geom_in_dim,
            hidden_dim=hidden_dim,
            out_dim=self.tp_rank,
            n_layers=3,
        )
        self.energy_coeff_mlp = MLP(
            in_dim=e_dim,
            hidden_dim=hidden_dim,
            out_dim=self.tp_rank,
            n_layers=3,
        )

        self.rank_basis = nn.Parameter(
            torch.randn(self.tp_rank, self.weight_numel) * (self.weight_numel ** -0.5)
        )

        self.score_mlp = MLP(
            in_dim=2 * self.inv_dim + z_emb_dim + rbf_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            n_layers=3,
        )

        self.out_mlp = MLP(
            in_dim=self.inv_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=3,
        )

    def forward(
        self,
        h_full: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        absorber_index: int = 0,
        geom: Optional[dict] = None,
    ) -> torch.Tensor:
        B, N, D = h_full.shape
        nE, dE = e_feat.shape
        dtype = h_full.dtype

        if geom is None:
            geom = build_absorber_relative_geometry(z, pos, mask, absorber_index)

        r = geom["r"]
        u = geom["u"]
        valid = geom["valid_neigh"] & (r <= self.cutoff)

        inv_abs = invariant_features_from_irreps(h_full[:, absorber_index, :], self.irreps_node)
        inv_nei = invariant_features_from_irreps(h_full, self.irreps_node)

        zr = self.z_emb(z)
        rr = self.rbf(r.clamp(max=self.cutoff))

        sh = o3.spherical_harmonics(
            self.irreps_sh,
            u.reshape(-1, 3),
            normalize=True,
            normalization="component",
        ).view(B, N, self.irreps_sh.dim)

        geom_in = torch.cat([zr, rr], dim=-1)
        base_w = self.base_weight_mlp(geom_in.reshape(B * N, -1)).view(B, N, self.weight_numel)
        geom_coeff = self.geom_coeff_mlp(geom_in.reshape(B * N, -1)).view(B, N, self.tp_rank)
        energy_coeff = self.energy_coeff_mlp(e_feat).unsqueeze(0).expand(B, nE, self.tp_rank)

        coeff = geom_coeff.unsqueeze(1) * energy_coeff.unsqueeze(2)   # [B,nE,N,R]
        delta_w = torch.einsum("benr,rw->benw", coeff, self.rank_basis)
        tp_w = base_w.unsqueeze(1) + delta_w  # [B,nE,N,W]

        h_rep = h_full.unsqueeze(1).expand(B, nE, N, D).reshape(B * nE * N, D)
        sh_rep = sh.unsqueeze(1).expand(B, nE, N, self.irreps_sh.dim).reshape(B * nE * N, self.irreps_sh.dim)
        tp_w_rep = tp_w.reshape(B * nE * N, self.weight_numel)

        values = self.value_tp(h_rep, sh_rep, tp_w_rep).view(B, nE, N, self.irreps_node.dim)

        score_in = torch.cat(
            [
                inv_abs.unsqueeze(1).unsqueeze(1).expand(B, nE, N, self.inv_dim),
                inv_nei.unsqueeze(1).expand(B, nE, N, self.inv_dim),
                zr.unsqueeze(1).expand(B, nE, N, zr.shape[-1]),
                rr.unsqueeze(1).expand(B, nE, N, rr.shape[-1]),
                e_feat.unsqueeze(0).unsqueeze(2).expand(B, nE, N, dE),
            ],
            dim=-1,
        )

        gate = torch.sigmoid(self.score_mlp(score_in).squeeze(-1))
        gate = gate * self.cutoff_fn(r).unsqueeze(1)
        gate = gate * valid.unsqueeze(1).to(dtype)

        agg = (gate.unsqueeze(-1) * values).sum(dim=2)
        norm = gate.sum(dim=2, keepdim=True).clamp_min(1e-8)
        agg = agg / norm

        inv_agg = invariant_features_from_irreps(agg, self.irreps_node)
        return self.out_mlp(inv_agg)


# ============================================================
# Multipole field blocks
# ============================================================


class InitialMultipoleProjector(nn.Module):
    """
    PySCF charges/spins are the baseline.
    The network predicts corrections to monopoles and initial dipoles.
    """
    def __init__(self, atom_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(
            in_dim=atom_dim + 2,
            hidden_dim=hidden_dim,
            out_dim=8,
            n_layers=3,
        )

    def forward(
        self,
        h: torch.Tensor,
        atom_charges: Optional[torch.Tensor] = None,
        atom_spins: Optional[torch.Tensor] = None,
    ):
        B, N, _ = h.shape
        dtype = h.dtype
        device = h.device

        if atom_charges is None:
            atom_charges = torch.zeros(B, N, device=device, dtype=dtype)
        if atom_spins is None:
            atom_spins = torch.zeros(B, N, device=device, dtype=dtype)

        x = torch.cat([h, atom_charges.unsqueeze(-1), atom_spins.unsqueeze(-1)], dim=-1)
        out = self.net(x)

        q0 = atom_charges + out[..., 0]
        m0 = atom_spins + out[..., 1]
        mu_q0 = out[..., 2:5]
        mu_m0 = out[..., 5:8]

        return q0, m0, mu_q0, mu_m0


class GaussianMultipoleFieldBuilder(nn.Module):
    def __init__(self, cutoff: float, sigma: float = 0.6, eps: float = 0.15):
        super().__init__()
        self.cutoff = float(cutoff)
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(
        self,
        pos: torch.Tensor,
        mask: torch.Tensor,
        q: torch.Tensor,
        m: torch.Tensor,
        mu_q: torch.Tensor,
        mu_m: torch.Tensor,
    ):
        rij = pos[:, :, None, :] - pos[:, None, :, :]
        dij = torch.linalg.norm(rij, dim=-1)
        uij = rij / dij.unsqueeze(-1).clamp_min(1e-8)

        valid = mask[:, :, None] & mask[:, None, :]
        not_self = dij > 1e-8
        pair_mask = valid & not_self & (dij <= self.cutoff)

        invr = 1.0 / torch.sqrt(dij ** 2 + self.eps ** 2)
        w_cut = self.cutoff_fn(dij) * pair_mask.to(pos.dtype)
        w_g = torch.exp(-0.5 * (dij / self.sigma) ** 2)
        w = w_cut * w_g

        qj = q[:, None, :]
        mj = m[:, None, :]

        vq_mono = (w * invr * qj).sum(dim=-1)
        vm_mono = (w * invr * mj).sum(dim=-1)

        eq_mono = ((w * invr * invr * qj).unsqueeze(-1) * uij).sum(dim=2)
        em_mono = ((w * invr * invr * mj).unsqueeze(-1) * uij).sum(dim=2)

        muqj = mu_q[:, None, :, :]
        mumj = mu_m[:, None, :, :]

        muq_dot_u = (muqj * uij).sum(dim=-1)
        mum_dot_u = (mumj * uij).sum(dim=-1)

        vq_dip = (w * invr * invr * muq_dot_u).sum(dim=-1)
        vm_dip = (w * invr * invr * mum_dot_u).sum(dim=-1)

        eq_dip = ((w * invr.pow(3) * muq_dot_u).unsqueeze(-1) * uij).sum(dim=2)
        em_dip = ((w * invr.pow(3) * mum_dot_u).unsqueeze(-1) * uij).sum(dim=2)

        return {
            "vq": vq_mono + vq_dip,
            "vm": vm_mono + vm_dip,
            "eq_vec": eq_mono + eq_dip,
            "em_vec": em_mono + em_dip,
        }


class MultipoleUpdater(nn.Module):
    def __init__(self, atom_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(
            in_dim=atom_dim + 2 + 6 + 2 + 6,
            hidden_dim=hidden_dim,
            out_dim=12,
            n_layers=3,
        )

    def forward(self, h, q, m, mu_q, mu_m, field):
        x = torch.cat(
            [
                h,
                q.unsqueeze(-1),
                m.unsqueeze(-1),
                mu_q,
                mu_m,
                field["vq"].unsqueeze(-1),
                field["vm"].unsqueeze(-1),
                field["eq_vec"],
                field["em_vec"],
            ],
            dim=-1,
        )
        out = self.net(x)

        dq = out[..., 0]
        dm = out[..., 1]
        dmu_q = out[..., 2:5]
        dmu_m = out[..., 5:8]
        fq = torch.nn.functional.softplus(out[..., 8]) + 1e-6
        fm = torch.nn.functional.softplus(out[..., 9]) + 1e-6
        alpha_q = torch.sigmoid(out[..., 10]).unsqueeze(-1)
        alpha_m = torch.sigmoid(out[..., 11]).unsqueeze(-1)

        return dq, dm, dmu_q, dmu_m, fq, fm, alpha_q, alpha_m


class ChargeSpinEquilibrator(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        m: torch.Tensor,
        fq: torch.Tensor,
        fm: torch.Tensor,
        mask: torch.Tensor,
        total_charge: Optional[torch.Tensor] = None,
        total_spin: Optional[torch.Tensor] = None,
    ):
        B, N = q.shape
        dtype = q.dtype
        device = q.device

        if total_charge is None:
            total_charge = torch.zeros(B, device=device, dtype=dtype)
        if total_spin is None:
            total_spin = torch.zeros(B, device=device, dtype=dtype)

        if total_charge.ndim > 1:
            total_charge = total_charge.view(total_charge.shape[0], -1).squeeze(-1)
        if total_spin.ndim > 1:
            total_spin = total_spin.view(total_spin.shape[0], -1).squeeze(-1)

        maskf = mask.to(dtype)

        q = q * maskf
        m = m * maskf
        fq = fq * maskf
        fm = fm * maskf

        dq = total_charge - q.sum(dim=1)
        dm = total_spin - m.sum(dim=1)

        fq_norm = fq / fq.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fm_norm = fm / fm.sum(dim=1, keepdim=True).clamp_min(1e-8)

        q = (q + fq_norm * dq.unsqueeze(-1)) * maskf
        m = (m + fm_norm * dm.unsqueeze(-1)) * maskf
        return q, m


class MultipoleFieldRefiner(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int,
        cutoff: float,
        n_iter: int = 2,
        sigma: float = 0.6,
    ):
        super().__init__()
        self.n_iter = n_iter

        self.init_proj = InitialMultipoleProjector(atom_dim=atom_dim, hidden_dim=hidden_dim)
        self.field_builder = GaussianMultipoleFieldBuilder(cutoff=cutoff, sigma=sigma)
        self.updater = MultipoleUpdater(atom_dim=atom_dim, hidden_dim=hidden_dim)
        self.equil = ChargeSpinEquilibrator()

        self.field_atom_summary = MLP(
            in_dim=2 + 6 + 2 + 6,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=3,
        )
        self.field_abs_summary = MLP(
            in_dim=2 + 6 + 2 + 6,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=3,
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        absorber_index: int = 0,
        atom_charges: Optional[torch.Tensor] = None,
        atom_spins: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        total_spin: Optional[torch.Tensor] = None,
    ):
        q, m, mu_q, mu_m = self.init_proj(
            h=h,
            atom_charges=atom_charges,
            atom_spins=atom_spins,
        )

        fq0 = torch.ones_like(q)
        fm0 = torch.ones_like(m)
        q, m = self.equil(q, m, fq0, fm0, mask, total_charge=total_charge, total_spin=total_spin)

        maskf = mask.unsqueeze(-1).to(h.dtype)
        mu_q = mu_q * maskf
        mu_m = mu_m * maskf

        for _ in range(self.n_iter):
            field = self.field_builder(pos, mask, q, m, mu_q, mu_m)
            dq, dm, dmu_q, dmu_m, fq, fm, alpha_q, alpha_m = self.updater(h, q, m, mu_q, mu_m, field)

            q = q + dq
            m = m + dm
            mu_q = mu_q + alpha_q * dmu_q
            mu_m = mu_m + alpha_m * dmu_m

            q, m = self.equil(q, m, fq, fm, mask, total_charge=total_charge, total_spin=total_spin)
            mu_q = mu_q * maskf
            mu_m = mu_m * maskf

        field = self.field_builder(pos, mask, q, m, mu_q, mu_m)

        atom_summary_in = torch.cat(
            [
                q.unsqueeze(-1),
                m.unsqueeze(-1),
                mu_q,
                mu_m,
                field["vq"].unsqueeze(-1),
                field["vm"].unsqueeze(-1),
                field["eq_vec"],
                field["em_vec"],
            ],
            dim=-1,
        )
        field_atom_lat = self.field_atom_summary(atom_summary_in)

        abs_summary_in = torch.cat(
            [
                q[:, absorber_index].unsqueeze(-1),
                m[:, absorber_index].unsqueeze(-1),
                mu_q[:, absorber_index, :],
                mu_m[:, absorber_index, :],
                field["vq"][:, absorber_index].unsqueeze(-1),
                field["vm"][:, absorber_index].unsqueeze(-1),
                field["eq_vec"][:, absorber_index, :],
                field["em_vec"][:, absorber_index, :],
            ],
            dim=-1,
        )
        field_abs_feat = self.field_abs_summary(abs_summary_in)

        return {
            "q": q,
            "m": m,
            "mu_q": mu_q,
            "mu_m": mu_m,
            "field": field,
            "field_atom_lat": field_atom_lat,
            "field_abs_feat": field_abs_feat,
        }


class EnergyConditionedFieldAttention(nn.Module):
    def __init__(self, field_dim: int, e_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.query = MLP(e_dim, hidden_dim, latent_dim, n_layers=3)
        self.key = MLP(field_dim, hidden_dim, latent_dim, n_layers=3)
        self.value = MLP(field_dim, hidden_dim, latent_dim, n_layers=3)
        self.out = MLP(latent_dim, hidden_dim, latent_dim, n_layers=2)
        self.scale = latent_dim ** -0.5

    def forward(self, field_atom_lat, mask, e_feat):
        B, N, _ = field_atom_lat.shape
        nE, _ = e_feat.shape

        q = self.query(e_feat).unsqueeze(0).expand(B, nE, -1)
        k = self.key(field_atom_lat)
        v = self.value(field_atom_lat)

        scores = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1) * self.scale
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        attn = torch.softmax(scores, dim=2)
        attn = attn * mask.unsqueeze(1).to(attn.dtype)

        out = (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=2)
        return self.out(out)


# ============================================================
# Final model
# ============================================================


@register_model("e3eenet")
@register_scheme("e3eenet", scheme_name="e3ee")
class E3EEmbed(Model):
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
        residual_scale_init: float = 0.1,
        attention_heads: int = 4,
        polar_hidden_dim: int = 128,
        polar_iterations: int = 2,
        field_sigma: float = 0.6,
        use_charge_spin: bool = True,
        use_invariant_block: bool = False,
        use_invariant_attention: bool = False,
        use_energy_dependent_equivariant_attention: bool = False,
        tp_rank: int = 8,
    ):
        super().__init__()
        self.nn_flag = 1
        self.gnn_flag = 0
        self.batch_flag = 1

        self.use_charge_spin = use_charge_spin
        self.use_invariant_block = use_invariant_block
        self.use_invariant_attention = use_invariant_attention
        self.use_energy_dependent_equivariant_attention = use_energy_dependent_equivariant_attention
        self.tp_rank = tp_rank

        if self.use_invariant_attention and self.use_energy_dependent_equivariant_attention:
            raise ValueError(
                "use_invariant_attention and use_energy_dependent_equivariant_attention "
                "cannot both be True."
            )

        self.energy_min = 0.0
        self.energy_max = float(out_features - 1)

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
            use_invariant_block=use_invariant_block,
        )

        self.inv_dim = invariant_feature_dim(self.atom_encoder.irreps_node)

        self.energy_embedding = EnergyRBFEmbedding(
            e_min=self.energy_min,
            e_max=self.energy_max,
            n_rbf=energy_rbf_dim,
        )

        if self.use_charge_spin:
            self.field_atom_dim = polar_hidden_dim
            self.field_abs_dim = polar_hidden_dim

            self.polar_refiner = MultipoleFieldRefiner(
                atom_dim=self.inv_dim,
                hidden_dim=polar_hidden_dim,
                cutoff=local_cutoff,
                n_iter=polar_iterations,
                sigma=field_sigma,
            )

            self.abs_branch = FieldConditionedAbsorberBranch(
                atom_dim=self.inv_dim,
                e_dim=energy_rbf_dim,
                field_abs_dim=self.field_abs_dim,
                hidden_dim=head_hidden_dim,
                out_dim=latent_dim,
            )

            if self.use_invariant_attention:
                self.atom_attention = FieldConditionedAtomAttention(
                    atom_dim=self.inv_dim,
                    e_dim=energy_rbf_dim,
                    rbf_dim=rbf_dim,
                    hidden_dim=atom_hidden_dim,
                    latent_dim=latent_dim,
                    cutoff=local_cutoff,
                    field_atom_dim=self.field_atom_dim,
                    field_abs_dim=self.field_abs_dim,
                    max_z=max_z,
                    z_emb_dim=32,
                    n_heads=attention_heads,
                )
            elif self.use_energy_dependent_equivariant_attention:
                self.atom_attention = FieldConditionedEquivariantAtomAttentionLowRank(
                    irreps_node=self.atom_encoder.irreps_node,
                    irreps_sh=self.atom_encoder.irreps_sh,
                    e_dim=energy_rbf_dim,
                    rbf_dim=rbf_dim,
                    hidden_dim=atom_hidden_dim,
                    latent_dim=latent_dim,
                    cutoff=local_cutoff,
                    field_atom_dim=self.field_atom_dim,
                    field_abs_dim=self.field_abs_dim,
                    tp_rank=tp_rank,
                    max_z=max_z,
                    z_emb_dim=32,
                )
            else:
                self.atom_attention = FieldConditionedEquivariantAtomAttention(
                    irreps_node=self.atom_encoder.irreps_node,
                    irreps_sh=self.atom_encoder.irreps_sh,
                    e_dim=energy_rbf_dim,
                    rbf_dim=rbf_dim,
                    hidden_dim=atom_hidden_dim,
                    latent_dim=latent_dim,
                    cutoff=local_cutoff,
                    field_atom_dim=self.field_atom_dim,
                    field_abs_dim=self.field_abs_dim,
                    max_z=max_z,
                    z_emb_dim=32,
                )

            self.eq_abs_head = FieldConditionedEquivariantAbsorberHead(
                irreps_node=self.atom_encoder.irreps_node,
                e_dim=energy_rbf_dim,
                field_abs_dim=self.field_abs_dim,
                hidden_dim=head_hidden_dim,
                out_dim=latent_dim,
            )

            self.field_attention = EnergyConditionedFieldAttention(
                field_dim=polar_hidden_dim,
                e_dim=energy_rbf_dim,
                hidden_dim=head_hidden_dim,
                latent_dim=latent_dim,
            )

            if self.use_invariant_block:
                local_in_dim = 2 * latent_dim
            else:
                local_in_dim = 3 * latent_dim
            self.local_head = MLP(
                in_dim=local_in_dim,
                hidden_dim=head_hidden_dim,
                out_dim=1,
                n_layers=out_mlp_layers,
            )

            field_corr_in_dim = 2 * latent_dim + self.field_abs_dim
            self.field_head = MLP(
                in_dim=field_corr_in_dim,
                hidden_dim=head_hidden_dim,
                out_dim=1,
                n_layers=out_mlp_layers,
            )

            self.field_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        else:
            self.abs_branch = EnergyConditionedAbsorberBranch(
                atom_dim=self.inv_dim,
                e_dim=energy_rbf_dim,
                hidden_dim=head_hidden_dim,
                out_dim=latent_dim,
            )

            if self.use_invariant_attention:
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
            elif self.use_energy_dependent_equivariant_attention:
                self.atom_attention = EnergyConditionedEquivariantAtomAttentionLowRank(
                    irreps_node=self.atom_encoder.irreps_node,
                    irreps_sh=self.atom_encoder.irreps_sh,
                    e_dim=energy_rbf_dim,
                    rbf_dim=rbf_dim,
                    hidden_dim=atom_hidden_dim,
                    latent_dim=latent_dim,
                    cutoff=local_cutoff,
                    tp_rank=tp_rank,
                    max_z=max_z,
                    z_emb_dim=32,
                )
            else:
                self.atom_attention = EnergyConditionedEquivariantAtomAttention(
                    irreps_node=self.atom_encoder.irreps_node,
                    irreps_sh=self.atom_encoder.irreps_sh,
                    e_dim=energy_rbf_dim,
                    rbf_dim=rbf_dim,
                    hidden_dim=atom_hidden_dim,
                    latent_dim=latent_dim,
                    cutoff=local_cutoff,
                    max_z=max_z,
                    z_emb_dim=32,
                )

            self.eq_abs_head = EnergyConditionedEquivariantAbsorberHead(
                irreps_node=self.atom_encoder.irreps_node,
                e_dim=energy_rbf_dim,
                hidden_dim=head_hidden_dim,
                out_dim=latent_dim,
            )

            if self.use_invariant_block:
                local_in_dim = 2 * latent_dim
            else:
                local_in_dim = 3 * latent_dim
            self.local_head = MLP(
                in_dim=local_in_dim,
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
                "residual_scale_init": residual_scale_init,
                "attention_heads": attention_heads,
                "polar_hidden_dim": polar_hidden_dim,
                "polar_iterations": polar_iterations,
                "field_sigma": field_sigma,
                "use_charge_spin": use_charge_spin,
                "use_invariant_block": use_invariant_block,
                "use_invariant_attention": use_invariant_attention,
                "use_energy_dependent_equivariant_attention": use_energy_dependent_equivariant_attention,
                "tp_rank": tp_rank,
            },
            type="e3eenet",
        )

    def _get_energy_grid(self, batch, device, dtype):
        return torch.arange(batch.y.shape[-1], device=device, dtype=dtype)

    def get_descriptor(self, batch):
        z = batch.z
        pos = batch.pos
        mask = batch.mask
        absorber_index = (
            int(batch.absorber_index.item())
            if hasattr(batch, "absorber_index") and torch.is_tensor(batch.absorber_index)
            else 0
        )

        geom = build_absorber_relative_geometry(
            z=z,
            pos=pos,
            mask=mask,
            absorber_index=absorber_index,
        )

        energies = self._get_energy_grid(batch, device=pos.device, dtype=pos.dtype)
        e_feat = self.energy_embedding(energies)

        h_full = self.atom_encoder(
            z=z,
            pos=pos,
            mask=mask,
            absorber_index=absorber_index,
            geom=geom,
        )
        h = invariant_features_from_irreps(h_full, self.atom_encoder.irreps_node)

        if self.use_charge_spin:
            atom_charges = getattr(batch, "atom_charges", None)
            atom_spins = getattr(batch, "atom_spins", None)
            total_charge = getattr(batch, "charge", None)
            total_spin = getattr(batch, "spin", None)

            polar = self.polar_refiner(
                h=h,
                pos=pos,
                mask=mask,
                absorber_index=absorber_index,
                atom_charges=atom_charges,
                atom_spins=atom_spins,
                total_charge=total_charge,
                total_spin=total_spin,
            )

            field_atom_lat = polar["field_atom_lat"]
            field_abs_feat = polar["field_abs_feat"]

            abs_lat = self.abs_branch(
                h[:, absorber_index, :],
                e_feat,
                field_abs_feat,
            )

            if self.use_invariant_attention:
                attn_lat = self.atom_attention(
                    h=h,
                    z=z,
                    pos=pos,
                    mask=mask,
                    e_feat=e_feat,
                    field_atom_lat=field_atom_lat,
                    field_abs_feat=field_abs_feat,
                    absorber_index=absorber_index,
                    geom=geom,
                )
            else:
                attn_lat = self.atom_attention(
                    h_full=h_full,
                    z=z,
                    pos=pos,
                    mask=mask,
                    e_feat=e_feat,
                    field_atom_lat=field_atom_lat,
                    field_abs_feat=field_abs_feat,
                    absorber_index=absorber_index,
                    geom=geom,
                )

            eq_abs_lat = self.eq_abs_head(
                h_full[:, absorber_index, :],
                e_feat,
                field_abs_feat,
            )

            field_lat = self.field_attention(
                field_atom_lat=field_atom_lat,
                mask=mask,
                e_feat=e_feat,
            )

            if self.use_invariant_block:
                local_parts = [eq_abs_lat, attn_lat]
            else:
                local_parts = [abs_lat, eq_abs_lat, attn_lat]
            local_x = torch.cat(local_parts, dim=-1)

            field_x = torch.cat(
                [
                    field_lat,
                    abs_lat,
                    field_abs_feat.unsqueeze(1).expand(
                        field_lat.shape[0], field_lat.shape[1], field_abs_feat.shape[-1]
                    ),
                ],
                dim=-1,
            )

            return {
                "local_x": local_x,
                "field_x": field_x,
                "use_charge_spin": True,
            }

        else:
            abs_lat = self.abs_branch(
                h[:, absorber_index, :],
                e_feat,
            )

            if self.use_invariant_attention:
                attn_lat = self.atom_attention(
                    h=h,
                    z=z,
                    pos=pos,
                    mask=mask,
                    e_feat=e_feat,
                    absorber_index=absorber_index,
                    geom=geom,
                )
            else:
                attn_lat = self.atom_attention(
                    h_full=h_full,
                    z=z,
                    pos=pos,
                    mask=mask,
                    e_feat=e_feat,
                    absorber_index=absorber_index,
                    geom=geom,
                )

            eq_abs_lat = self.eq_abs_head(
                h_full[:, absorber_index, :],
                e_feat,
            )

            if self.use_invariant_block:
                local_parts = [eq_abs_lat, attn_lat]
            else:
                local_parts = [abs_lat, eq_abs_lat, attn_lat]
            local_x = torch.cat(local_parts, dim=-1)

            return {
                "local_x": local_x,
                "use_charge_spin": False,
            }

    def forward(self, batch):
        desc = self.get_descriptor(batch)

        y_local = self.local_head(desc["local_x"]).squeeze(-1)

        if desc["use_charge_spin"]:
            y_field = self.field_head(desc["field_x"]).squeeze(-1)
            y_pred = y_local + self.field_scale.to(y_local.dtype) * y_field
        else:
            y_pred = y_local

        return y_pred
