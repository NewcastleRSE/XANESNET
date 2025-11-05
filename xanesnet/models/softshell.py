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

from typing import List

import torch

from torch import nn

from xanesnet.registry import register_model, register_scheme
from xanesnet.models.base_model import Model


@register_model("softshell")
@register_scheme("softshell", scheme_name="ss")
class SoftShellSpectraNet(Model):
    """
    Wrapper class for SoftShell Model
    Structure Encoder + Coefficient Head
    """

    def __init__(
        self,
        in_size: List,  # descriptor feature + K_group
        out_size: int,
        n_shells: int = 4,
        latent_dim: int = 512,
        max_radius_angs: float = 7.0,
        init_width: float = 0.8,
        use_gating: bool = True,
        head_hidden: int = 256,
        head_depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.nn_flag = 1
        self.batch_flag = 1

        # Save model configuration
        self.register_config(locals(), type="softshell")

        d_input = in_size[0]  # descriptor feature size
        kgroups = in_size[1]  # K-groups

        self.encoder = SoftRadialShellsEncoder(
            d_input=d_input,
            n_shells=n_shells,
            latent_dim=latent_dim,
            max_radius_angs=max_radius_angs,
            init_centers=None,
            init_width=init_width,
            use_gating=use_gating,
        )

        self.coeff_head = CoeffHeadGroupedResidualPreLN(
            latent_dim=latent_dim,
            K_groups=kgroups,
            hidden=head_hidden,
            depth=head_depth,
            dropout=dropout,
        )

    def forward(self, batch):
        h = self.encoder(batch.desc, lengths=batch.lengths, dists=batch.dist)
        return self.coeff_head(h)

    def forward_encoder(self, x, lengths=None, dists=None):
        return self.encoder(x, lengths=lengths, dists=dists)

    def forward_coeffs(self, latent):
        return self.coeff_head(latent)


class SpectralBasis(nn.Module):
    def __init__(
        self,
        energies: torch.Tensor,
        widths_bins=(1.0, 3.0, 7.0),
        normalize_atoms=True,
        stride=1,
    ):
        super().__init__()
        self.register_buffer("E", energies.detach().clone())
        self.widths_bins = [float(w) for w in widths_bins]
        self.normalize_atoms = bool(normalize_atoms)
        self.stride = int(stride)

        N = energies.numel()
        dE = float(energies[1] - energies[0])

        grid_idx = torch.arange(N, device=energies.device, dtype=energies.dtype)
        centers_grid = grid_idx[:: self.stride]
        diff_bins = grid_idx.unsqueeze(1) - centers_grid.unsqueeze(0)

        Phi_list, centers_list = [], []
        for w in self.widths_bins:
            Phi_w = torch.exp(-0.5 * (diff_bins / w) ** 2)  # (N, n_centers_per_width)
            Phi_list.append(Phi_w)
            centers_list.append(self.E[:: self.stride])

        Phi = torch.cat(Phi_list, dim=1)  # (N, K_no_const)
        centers = torch.cat(centers_list)  # (K_no_const,)

        if self.normalize_atoms:
            Phi = Phi / (Phi.sum(dim=0, keepdim=True) * dE + 1e-12)

        self.register_buffer("Phi", Phi)  # (N, K)
        self.register_buffer("centers", centers)  # (K,)

    def synthesize(self, coeffs: torch.Tensor):
        return coeffs @ self.Phi.T  # (B, N)


class SpectralPost(nn.Module):
    """
    Stage 1: parameter-free (no training).
      y = Φ c    (optionally clamped to nonnegative)
    """

    def __init__(self, basis: "SpectralBasis", nonneg_output: bool = False):
        super().__init__()
        self.basis = basis
        self.nonneg_output = bool(nonneg_output)

    def forward_from_coeffs(self, coeffs: torch.Tensor):
        y = self.basis.synthesize(coeffs)
        if self.nonneg_output:
            y = y.clamp_min_(0)
        return y


class SoftRadialShellsEncoder(nn.Module):
    """
    Absorber-centric soft-binning over distance with learnable shell centers/widths.
    Inputs:
      x: (B, N, H)  with absorber at index 0
      dists: (B, N)
    Output:
      (B, latent_dim)
    """

    def __init__(
        self,
        d_input=256,
        n_shells=4,
        latent_dim=512,
        max_radius_angs=7.0,
        init_centers=None,
        init_width=0.8,
        use_gating=True,
    ):
        super().__init__()
        self.max_radius = float(max_radius_angs)
        self.n_shells = int(n_shells)
        self.d_input = int(d_input)
        self.latent_dim = int(latent_dim)

        # Initialize shell centers (roughly equally spaced) and widths
        if init_centers is None:
            centers = torch.linspace(0.5, self.max_radius - 0.5, steps=self.n_shells)
        else:
            centers = torch.as_tensor(init_centers, dtype=torch.float32)
            assert centers.numel() == self.n_shells
        widths = torch.full((self.n_shells,), float(init_width))

        self.shell_centers = nn.Parameter(centers)  # (S,)
        self.shell_widths = nn.Parameter(widths.clamp_min(1e-2))  # (S,)

        # self.post_shell = nn.Sequential(
        #     nn.Linear(d_input * self.n_shells, 2 * d_input),
        #     nn.GELU(),
        #     nn.Linear(2 * d_input, d_input),
        #     nn.GELU(),
        # )

        self.post_shell = nn.Sequential(
            nn.Linear(d_input * self.n_shells, d_input),
        )

        self.use_gating = bool(use_gating)
        if self.use_gating:
            # Distance-aware gate to modulate shell summary before concat
            self.gate = nn.Sequential(
                nn.Linear(d_input + 16, d_input),
                nn.GELU(),
                nn.Linear(d_input, d_input),
                nn.Sigmoid(),
            )
            # Simple Fourier features for absorber’s local crowding
            self.register_buffer(
                "freqs", torch.linspace(0.5, 6.0, 8)
            )  # 8 freqs → 16 features after sin/cos

        self.fuse = nn.Sequential(
            nn.Linear(d_input * 2, 2 * d_input),
            nn.GELU(),
            nn.Linear(2 * d_input, latent_dim),
        )
        self.apply(init_mlp_weights)

    def _soft_assign(self, r):  # r: (B, N-1)
        centers = self.shell_centers.view(1, 1, -1)  # (1,1,S)
        widths = self.shell_widths.view(1, 1, -1)  # (1,1,S)
        z = (r.unsqueeze(-1) - centers) / (widths + 1e-6)  # (B, N-1, S)
        w = torch.exp(-0.5 * z * z)  # (B, N-1, S)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)  # normalize across atoms
        return w  # (B, N-1, S)

    def _fourier_feats(self, r):  # r: (B, N-1)
        f = self.freqs.view(1, 1, -1)  # (1,1,F)
        fsin = torch.sin(r.unsqueeze(-1) * f)  # (B,N-1,F)
        fcos = torch.cos(r.unsqueeze(-1) * f)  # (B,N-1,F)
        return torch.cat([fsin, fcos], dim=-1).mean(
            dim=1
        )  # (B,2F) crowding summary = (B,16)

    def forward(self, x, lengths=None, dists=None):
        assert dists is not None, "SoftRadialShellsEncoder requires dists."
        B, N, H = x.shape
        absorbing = x[:, 0, :]  # (B,H)
        context = x[:, 1:, :]  # (B,N-1,H)
        r = dists[:, 1:].clamp_max(self.max_radius)  # (B,N-1)

        # Valid mask
        if lengths is not None:
            n_ctx = context.size(1)
            idxs = torch.arange(n_ctx, device=x.device)[None, :]
            real_ctx = torch.clamp(lengths - 1, min=0)
            mask = (idxs < real_ctx[:, None]).float()
        else:
            mask = torch.ones(context.shape[:2], device=x.device)
        mask = mask * (r <= self.max_radius).float()  # (B,N-1)

        # Soft shell assignment & renormalize over valid atoms
        w = self._soft_assign(r)  # (B,N-1,S)
        w = w * mask.unsqueeze(-1)  # zero invalid
        wsum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w = w / wsum

        # Weighted shell means
        shell_means = torch.einsum("bns,bnh->bsh", w, context)  # (B,S,H)
        shell_means = shell_means.reshape(B, self.n_shells * H)  # (B,S*H)
        shell_summary = self.post_shell(shell_means)  # (B,H)

        if self.use_gating:
            crowd = self._fourier_feats(r)  # (B,16)
            gate_in = torch.cat([absorbing, crowd], dim=-1)  # (B,H+16)
            g = self.gate(gate_in)  # (B,H) in [0,1]
            shell_summary = shell_summary * g

        fused = torch.cat([absorbing, shell_summary], dim=-1)
        return fused


# -------------------------
# NEW Coeff Head: Grouped + Residual + Pre-LN
# -------------------------
class ResidualPreLNBlock(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        init_mlp_weights(self.fc1)
        init_mlp_weights(self.fc2)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class CoeffHeadGroupedResidualPreLN(nn.Module):
    """
    Shared residual Pre-LN trunk over latent; per-width grouped linear heads.
    If a constant column is used in the basis, an extra 1-d head is appended automatically.
    """

    def __init__(
        self,
        latent_dim: int,
        K_groups: List,
        hidden=256,
        depth=3,
        dropout=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.K_groups = K_groups
        self.trunk = nn.Sequential(
            *[ResidualPreLNBlock(latent_dim, hidden, dropout) for _ in range(depth)]
        )
        self.trunk_out_ln = nn.LayerNorm(latent_dim)
        self.group_heads = nn.ModuleList(
            [nn.Linear(latent_dim, k) for k in self.K_groups]
        )
        # Initialize group heads for stable early training
        for head in self.group_heads:
            # init_mlp_weights(head)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, z):
        h = self.trunk(z)
        h = self.trunk_out_ln(h)
        outs = [head(h) for head in self.group_heads]
        return torch.cat(outs, dim=-1)  # (B, sum K_groups)


def init_mlp_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
