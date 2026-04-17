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

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct

# ============================================================
# Utility functions
# ============================================================


def build_absorber_relative_geometry(
    pos: torch.Tensor,
    mask: torch.Tensor,
    absorber_index: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Build absorber-relative geometry once and reuse it everywhere.

    Args:
        pos:            [B, N, 3] atom positions
        mask:           [B, N] valid-atom mask
        absorber_index: index of the absorber atom (default 0)

    Returns dict with:
        rel         [B, N, 3]  position relative to absorber
        r           [B, N]     absorber-neighbour distance
        u           [B, N, 3]  absorber-neighbour unit vector
        valid_neigh [B, N]     valid neighbours excluding absorber
    """
    abs_pos = pos[:, absorber_index, :].unsqueeze(1)  # [B, 1, 3]
    rel = pos - abs_pos  # [B, N, 3]
    r = torch.linalg.norm(rel, dim=-1)  # [B, N]
    u = rel / r.unsqueeze(-1).clamp_min(1e-8)  # [B, N, 3]

    valid_neigh = mask.clone()
    valid_neigh[:, absorber_index] = False

    return {
        "rel": rel,
        "r": r,
        "u": u,
        "valid_neigh": valid_neigh,
    }


def invariant_feature_dim(irreps: o3.Irreps) -> int:
    """Number of invariant scalar channels extractable from *irreps*."""
    return sum(mul for mul, _ in irreps)


def invariant_features_from_irreps(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """
    Convert flattened irreps features to invariant features.

    For l = 0: keep scalar channels directly.
    For l > 0: take RMS norm of each irrep copy.

    Args:
        x: [..., D] tensor of concatenated irreps features
        irreps: the irreps specification matching the last dimension of x

    Returns:
        [..., inv_dim] tensor of invariant features
    """
    orig_shape = x.shape[:-1]
    D = x.shape[-1]
    x_flat = x.reshape(-1, D)
    M = x_flat.shape[0]

    outs: list[torch.Tensor] = []
    offset = 0
    for mul, ir in irreps:
        dim = ir.dim
        block_dim = mul * dim
        xb = x_flat[:, offset : offset + block_dim].reshape(M, mul, dim)

        if ir.l == 0:
            outs.append(xb.reshape(M, mul))
        else:
            inv = torch.sqrt((xb**2).mean(dim=-1) + 1e-8)
            outs.append(inv)

        offset += block_dim

    out = torch.cat(outs, dim=-1)
    return out.view(*orig_shape, out.shape[-1])


# ============================================================
# Basic modules
# ============================================================


class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion."""

    def __init__(self, start: float, stop: float, n_rbf: int, gamma: float | None = None) -> None:
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
    """Smooth cosine cutoff envelope."""

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r / self.cutoff
        out = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        out = out * (r <= self.cutoff).to(r.dtype)
        return out


class EnergyRBFEmbedding(nn.Module):
    """Gaussian RBF embedding for scalar energy values."""

    def __init__(self, e_min: float, e_max: float, n_rbf: int) -> None:
        super().__init__()
        self.rbf = GaussianRBF(e_min, e_max, n_rbf)

    def forward(self, energies: torch.Tensor) -> torch.Tensor:
        return self.rbf(energies)


class MLP(nn.Module):
    """Simple multi-layer perceptron with SiLU activations."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RadialMLP(nn.Module):
    """Two-hidden-layer MLP used for radial weight prediction."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IrrepNorm(nn.Module):
    """
    Irrep-respecting normalization.

    l = 0 blocks: mean/variance normalization across multiplicity channels.
    l > 0 blocks: RMS normalization per irrep copy.
    """

    def __init__(self, irreps: o3.Irreps, eps: float = 1e-8, affine: bool = True) -> None:
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

        outs: list[torch.Tensor] = []
        offset = 0
        Bflat = x.shape[0]

        for mul, ir in self.irreps:
            dim = ir.dim
            block_dim = mul * dim
            xb = x[:, offset : offset + block_dim].reshape(Bflat, mul, dim)

            if ir.l == 0:
                mean = xb.mean(dim=1, keepdim=True)
                var = ((xb - mean) ** 2).mean(dim=1, keepdim=True)
                xb = (xb - mean) / torch.sqrt(var + self.eps)
            else:
                norm = torch.sqrt((xb**2).mean(dim=2, keepdim=True) + self.eps)
                xb = xb / norm

            outs.append(xb.reshape(Bflat, block_dim))
            offset += block_dim

        out = torch.cat(outs, dim=-1)

        if self.affine:
            out = out * self.weight + self.bias  # TODO bias equivariance breaking?

        return out.view(*orig_shape, D)


# ============================================================
# Batched graph builder
# ============================================================


class BatchedRadiusGraphBuilder(nn.Module):
    """
    Vectorized padded-batch radius graph construction.

    Returns flattened node indices for a [B, N, ...] tensor layout,
    where flat index = b * N + i.
    """

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, pos: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = pos.device
        dtype = pos.dtype
        B, N, _ = pos.shape

        diff = pos[:, :, None, :] - pos[:, None, :, :]  # [B, N, N, 3]
        dist = torch.linalg.norm(diff, dim=-1)  # [B, N, N]

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
# Equivariant encoder components
# ============================================================


class EquivariantInteractionBlock(nn.Module):
    """
    Single equivariant message-passing interaction block using
    e3nn tensor products with gating.
    """

    def __init__(
        self,
        irreps_node: str,
        irreps_sh: str,
        irreps_message: str,
        rbf_dim: int,
        radial_hidden_dim: int,
        cutoff: float,
        residual_scale_init: float = 0.1,
    ) -> None:
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
        irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e") if irreps_gated.num_irreps > 0 else o3.Irreps("")

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

        norm = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        norm.index_add_(0, edge_dst, edge_w)
        agg = agg / norm.clamp_min(1e-8)

        agg = self.msg_linear(agg)
        agg = self.gate(agg)

        out = self.self_linear(x_norm) + self.update_linear(agg)
        return x + self.res_scale.to(x.dtype) * out


class EquivariantAtomEncoder(nn.Module):
    """
    Equivariant atom encoder with spherical harmonics message passing.

    Produces per-atom features in the given irreps representation.
    """

    def __init__(
        self,
        max_z: int,
        cutoff: float,
        num_interactions: int,
        rbf_dim: int,
        lmax: int,
        node_attr_dim: int,
        hidden_dim: int,
        irreps_node: str,
        irreps_message: str,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.rbf_dim = rbf_dim
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

        self.blocks = nn.ModuleList(
            [
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
            ]
        )

        self.out_norm = IrrepNorm(self.irreps_node)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        absorber_index: int = 0,
        geom: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:    [B, N] atomic numbers
            pos:  [B, N, 3] positions
            mask: [B, N] valid-atom mask
            absorber_index: index of the absorber atom
            geom: precomputed absorber-relative geometry (optional)

        Returns:
            [B, N, irreps_dim] equivariant atom features
        """
        device = pos.device
        B, N = z.shape

        if geom is None:
            geom = build_absorber_relative_geometry(pos=pos, mask=mask, absorber_index=absorber_index)

        r_abs = geom["r"]

        abs_flag = torch.zeros_like(z, dtype=pos.dtype)
        abs_flag[:, absorber_index] = 1.0

        # atomic numbers embedding
        zf = self.z_emb(z)
        # distance embedding (Gaussian RBF)
        # TODO: This is absorber specific. Could we not remove this?
        rf = self.dist_rbf(r_abs.clamp(max=self.cutoff))

        # [B, N, node_attr_dim + 1 + rbf_dim] concatenated scalar features for each atom
        scalar_in = torch.cat([zf, abs_flag.unsqueeze(-1), rf], dim=-1)
        # equivariant linear layer: [B * N, irreps_dim]
        x = self.input_lin(scalar_in.reshape(B * N, self.input_scalar_dim))
        # mask out padded atoms to zero
        flat_mask = mask.reshape(B * N)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)

        # build simple radius graph
        edge_src, edge_dst, edge_vec = self.graph_builder(pos, mask)

        if edge_vec.numel() > 0:  # handle case of no edges in batch
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
    """Energy-conditioned element-pair scattering features."""

    def __init__(
        self,
        max_z: int,
        z_emb_dim: int,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.mlp = MLP(
            in_dim=2 * z_emb_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(
        self,
        z_j: torch.Tensor,
        z_k: torch.Tensor,
        e_feat: torch.Tensor,
    ) -> torch.Tensor:
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
    ) -> None:
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

    def _enumerate_paths(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        absorber_index: int,
        geom: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = z.shape
        device = z.device

        if geom is None:
            geom = build_absorber_relative_geometry(pos=pos, mask=mask, absorber_index=absorber_index)

        r = geom["r"]
        valid = geom["valid_neigh"] & (r <= self.cutoff)

        all_j: list[torch.Tensor] = []
        all_k: list[torch.Tensor] = []
        all_m: list[torch.Tensor] = []
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

        return (
            torch.stack(all_j, dim=0),
            torch.stack(all_k, dim=0),
            torch.stack(all_m, dim=0),
        )

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        pair_elem_energy: PairElementEnergyScattering,
        e_feat: torch.Tensor,
        absorber_index: int = 0,
        geom: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, N, H = h.shape
        device = h.device

        if geom is None:
            geom = build_absorber_relative_geometry(pos=pos, mask=mask, absorber_index=absorber_index)

        j_idx, k_idx, path_mask = self._enumerate_paths(
            z=z, pos=pos, mask=mask, absorber_index=absorber_index, geom=geom
        )
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

        cutoff_w = (self.cutoff_fn(r0j) * self.cutoff_fn(r0k) * self.cutoff_fn(rjk)).unsqueeze(-1).unsqueeze(-1)

        contrib = g_geom.unsqueeze(2) * g_elem
        contrib = contrib * cutoff_w
        contrib = contrib * path_mask.unsqueeze(-1).unsqueeze(-1).to(contrib.dtype)

        agg = contrib.sum(dim=1)

        norm = cutoff_w.squeeze(-1) * path_mask.unsqueeze(-1).to(cutoff_w.dtype)
        norm = norm.sum(dim=1).clamp_min(1e-8)
        agg = agg / norm.unsqueeze(1)

        return self.out_proj(agg)


# ============================================================
# Energy-conditioned attention
# ============================================================


class EnergyConditionedAtomAttention(nn.Module):
    """
    Faster energy-conditioned attention over atomwise invariant features.

    Queries are energy-dependent; keys/values are atom-dependent but
    energy-independent.
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
    ) -> None:
        super().__init__()
        if latent_dim % n_heads != 0:
            raise ValueError("latent_dim must be divisible by n_heads")

        self.atom_dim = atom_dim
        self.e_dim = e_dim
        self.rbf_dim = rbf_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cutoff = cutoff
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads

        self.rbf_fn = GaussianRBF(0.0, cutoff, rbf_dim)
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

        self.score_scale = self.head_dim**-0.5

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
        geom: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h:      [B, N, H] invariant atom features
            z:      [B, N] atomic numbers
            pos:    [B, N, 3] positions
            mask:   [B, N] valid-atom mask
            e_feat: [nE, dE] energy feature embedding
            absorber_index: absorber atom index
            geom:   precomputed absorber-relative geometry

        Returns:
            [B, nE, latent_dim]
        """
        B, N, H = h.shape
        nE, dE = e_feat.shape

        if geom is None:
            geom = build_absorber_relative_geometry(pos=pos, mask=mask, absorber_index=absorber_index)

        r = geom["r"]  # [B, N]
        u = geom["u"]  # [B, N, 3]
        valid = mask & (r <= self.cutoff)

        # Query: absorber invariant features + energy embedding
        h_abs = h[:, absorber_index, :]  # [B, H]
        q_in = torch.cat(
            [
                h_abs.unsqueeze(1).expand(B, nE, H),
                e_feat.unsqueeze(0).expand(B, nE, dE),
            ],
            dim=-1,
        )
        q = self.query_mlp(q_in)  # [B, nE, L]

        # Static atom features
        zr = self.z_emb(z)  # [B, N, z_emb_dim]
        rr = self.rbf_fn(r.clamp(max=self.cutoff))  # [B, N, rbf_dim]
        is_abs = torch.zeros_like(r)
        is_abs[:, absorber_index] = 1.0

        atom_static = torch.cat([h, zr, rr, u, is_abs.unsqueeze(-1)], dim=-1)
        k = self.key_mlp(atom_static)  # [B, N, L]
        v = self.value_mlp(atom_static)  # [B, N, L]

        # Split into heads
        q = self._split_heads(q)  # [B, nE, nH, dH]
        k = self._split_heads(k)  # [B, N,  nH, dH]
        v = self._split_heads(v)  # [B, N,  nH, dH]

        # Scores: [B, nE, N, nH]
        scores = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1) * self.score_scale

        radial_bias = torch.log(self.cutoff_fn(r).clamp_min(1e-8)).unsqueeze(1).unsqueeze(-1)
        scores = scores + radial_bias

        attn_mask = valid.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~attn_mask, -1e9)

        attn = torch.softmax(scores, dim=2)
        attn = attn * attn_mask.to(attn.dtype)

        # Weighted sum over atoms → [B, nE, nH, dH]
        out = (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=2)
        out = out.reshape(B, nE, self.latent_dim)

        return self.out_proj(out)


# ============================================================
# Head branches
# ============================================================


class EnergyConditionedAbsorberBranch(nn.Module):
    """Energy-dependent absorber branch based on invariant absorber features."""

    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=atom_dim + e_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, h_abs: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_abs:  [B, H] absorber invariant features
            e_feat: [nE, dE] energy features

        Returns:
            [B, nE, out_dim]
        """
        B, H = h_abs.shape
        nE, dE = e_feat.shape

        ha = h_abs.unsqueeze(1).expand(B, nE, H)
        ef = e_feat.unsqueeze(0).expand(B, nE, dE)
        return self.mlp(torch.cat([ha, ef], dim=-1))


class EnergyIrrepModulation(nn.Module):
    """
    Energy-conditioned scalar modulation of each irrep copy,
    preserving equivariance.
    """

    def __init__(self, irreps: o3.Irreps, e_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.n_copies = sum(mul for mul, _ in self.irreps)

        self.mlp = MLP(
            in_dim=e_dim,
            hidden_dim=hidden_dim,
            out_dim=self.n_copies,
            n_layers=3,
        )

    def forward(self, x: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      [B, D] equivariant features
            e_feat: [nE, e_dim] energy features

        Returns:
            [B, nE, D]
        """
        B, D = x.shape
        nE = e_feat.shape[0]

        gates = self.mlp(e_feat)  # [nE, n_copies]

        outs: list[torch.Tensor] = []
        xoff = 0
        goff = 0

        for mul, ir in self.irreps:
            dim = ir.dim
            block_dim = mul * dim

            xb = x[:, xoff : xoff + block_dim].view(B, mul, dim)
            gb = gates[:, goff : goff + mul]

            xb = xb.unsqueeze(1)  # [B, 1, mul, dim]
            gb = gb.unsqueeze(0).unsqueeze(-1)  # [1, nE, mul, 1]

            outs.append((xb * gb).reshape(B, nE, block_dim))

            xoff += block_dim
            goff += mul

        return torch.cat(outs, dim=-1)


class EnergyConditionedEquivariantAbsorberHead(nn.Module):
    """
    Late equivariant absorber head.

    Applies energy-conditioned irrep-wise modulation to the absorber
    equivariant feature, converts to invariants, then projects.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        e_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.mod = EnergyIrrepModulation(self.irreps_node, e_dim=e_dim, hidden_dim=hidden_dim)
        self.inv_dim = invariant_feature_dim(self.irreps_node)

        self.out_mlp = MLP(
            in_dim=self.inv_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=3,
        )

    def forward(self, h_abs_full: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_abs_full: [B, D] equivariant absorber features
            e_feat:     [nE, e_dim] energy features

        Returns:
            [B, nE, out_dim]
        """
        h_mod = self.mod(h_abs_full, e_feat)  # [B, nE, D]
        inv = invariant_features_from_irreps(h_mod, self.irreps_node)  # [B, nE, inv_dim]
        return self.out_mlp(inv)
