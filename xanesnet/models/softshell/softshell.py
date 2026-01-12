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

from xanesnet.models.model import Model
from xanesnet.registry import ModelRegistry, SchemeRegistry


@ModelRegistry.register("softshell")
@SchemeRegistry.register("softshell", scheme_name="ss")
class SoftShellSpectraNet(Model):
    """
    Wrapper class for SoftShell Model
    Structure Encoder + Coefficient Head
    """

    def __init__(
        self,
        in_size: List,
        out_size: int,
        n_shells: int = 4,
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

        d_input = in_size[0]
        latent_dim = d_input * 2
        kgroups = in_size[1]

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

    def init_layer_weights(self, m, kernel_init_fn, bias_init_fn):
        """
        Initialise weights and bias for a single layer.
        Overrides base method to handle ResidualPreLNBlock.
        """
        # initialise layers in encoder
        if m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                kernel_init_fn(m.weight)
                bias_init_fn(m.bias)

        # initialise layers in coeff_head
        if m in self.coeff_head.modules():
            if isinstance(m, ResidualPreLNBlock):
                # initialize fc1 and fc2 inside the block
                kernel_init_fn(m.fc1.weight)
                bias_init_fn(m.fc1.bias)
                kernel_init_fn(m.fc2.weight)
                bias_init_fn(m.fc2.bias)


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

        if init_centers is None:
            centers = torch.linspace(0.5, self.max_radius - 0.5, steps=self.n_shells)
        else:
            centers = torch.as_tensor(init_centers, dtype=torch.float32)
            assert centers.numel() == self.n_shells
        widths = torch.full((self.n_shells,), float(init_width))

        self.shell_centers = nn.Parameter(centers)
        self.shell_widths = nn.Parameter(widths.clamp_min(1e-2))

        self.post_shell = nn.Sequential(
            nn.Linear(d_input * self.n_shells, d_input),
        )

        self.use_gating = bool(use_gating)
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(d_input + 16, d_input),
                nn.GELU(),
                nn.Linear(d_input, d_input),
                nn.Sigmoid(),
            )
            self.register_buffer("freqs", torch.linspace(0.5, 6.0, 8))

        self.fuse = nn.Sequential(
            nn.Linear(d_input * 2, 2 * d_input),
            nn.GELU(),
            nn.Linear(2 * d_input, latent_dim),
        )
        self.apply(init_mlp_weights)

    def _soft_assign(self, r):
        centers = self.shell_centers.view(1, 1, -1)
        widths = self.shell_widths.view(1, 1, -1)
        z = (r.unsqueeze(-1) - centers) / (widths + 1e-6)
        w = torch.exp(-0.5 * z * z)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        return w

    def _fourier_feats(self, r):
        f = self.freqs.view(1, 1, -1)
        fsin = torch.sin(r.unsqueeze(-1) * f)
        fcos = torch.cos(r.unsqueeze(-1) * f)
        return torch.cat([fsin, fcos], dim=-1).mean(dim=1)

    def forward(self, x, lengths=None, dists=None):
        assert dists is not None, "SoftRadialShellsEncoder requires dists."
        B, N, H = x.shape
        absorbing = x[:, 0, :]
        context = x[:, 1:, :]
        r = dists[:, 1:].clamp_max(self.max_radius)

        if lengths is not None:
            n_ctx = context.size(1)
            idxs = torch.arange(n_ctx, device=x.device)[None, :]
            real_ctx = torch.clamp(lengths - 1, min=0)
            mask = (idxs < real_ctx[:, None]).float()
        else:
            mask = torch.ones(context.shape[:2], device=x.device)
        mask = mask * (r <= self.max_radius).float()

        w = self._soft_assign(r)
        w = w * mask.unsqueeze(-1)
        wsum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w = w / wsum

        shell_means = torch.einsum("bns,bnh->bsh", w, context)
        shell_means = shell_means.reshape(B, self.n_shells * H)
        shell_summary = self.post_shell(shell_means)

        if self.use_gating:
            crowd = self._fourier_feats(r)
            gate_in = torch.cat([absorbing, crowd], dim=-1)
            g = self.gate(gate_in)
            shell_summary = shell_summary * g

        fused = torch.cat([absorbing, shell_summary], dim=-1)
        return fused


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
        self.trunk = nn.Sequential(*[ResidualPreLNBlock(latent_dim, hidden, dropout) for _ in range(depth)])
        self.trunk_out_ln = nn.LayerNorm(latent_dim)
        self.group_heads = nn.ModuleList([nn.Linear(latent_dim, k) for k in self.K_groups])

        for head in self.group_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, z):
        h = self.trunk(z)
        h = self.trunk_out_ln(h)
        outs = [head(h) for head in self.group_heads]
        return torch.cat(outs, dim=-1)


def init_mlp_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
