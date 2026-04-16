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

from ..utils import build_absorber_relative_geometry
from .basic import MLP, CosineCutoff, GaussianRBF


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
        bsz, n_paths = z_j.shape
        n_energies, e_dim = e_feat.shape

        ej = self.z_emb(z_j).unsqueeze(2).expand(bsz, n_paths, n_energies, -1)
        ek = self.z_emb(z_k).unsqueeze(2).expand(bsz, n_paths, n_energies, -1)
        ef = e_feat.unsqueeze(0).unsqueeze(0).expand(bsz, n_paths, n_energies, e_dim)

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
        bsz, _ = z.shape
        device = z.device

        if geom is None:
            geom = build_absorber_relative_geometry(pos=pos, mask=mask, absorber_index=absorber_index)

        r = geom["r"]
        valid = geom["valid_neigh"] & (r <= self.cutoff)

        all_j: list[torch.Tensor] = []
        all_k: list[torch.Tensor] = []
        all_m: list[torch.Tensor] = []
        pmax = self.max_paths_per_structure

        for b in range(bsz):
            idx = torch.where(valid[b])[0]

            if idx.numel() < 2:
                j_idx = torch.zeros(pmax, dtype=torch.long, device=device)
                k_idx = torch.zeros(pmax, dtype=torch.long, device=device)
                pmask = torch.zeros(pmax, dtype=torch.bool, device=device)
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

                if pairs.shape[0] > pmax:
                    pairs = pairs[:pmax]

                n_pairs = pairs.shape[0]
                j_idx = torch.zeros(pmax, dtype=torch.long, device=device)
                k_idx = torch.zeros(pmax, dtype=torch.long, device=device)
                pmask = torch.zeros(pmax, dtype=torch.bool, device=device)

                j_idx[:n_pairs] = pairs[:, 0]
                k_idx[:n_pairs] = pairs[:, 1]
                pmask[:n_pairs] = True

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
        bsz, _, _ = h.shape
        device = h.device

        if geom is None:
            geom = build_absorber_relative_geometry(pos=pos, mask=mask, absorber_index=absorber_index)

        j_idx, k_idx, path_mask = self._enumerate_paths(
            z=z, pos=pos, mask=mask, absorber_index=absorber_index, geom=geom
        )
        batch_idx = torch.arange(bsz, device=device)[:, None]

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
