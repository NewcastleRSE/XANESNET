# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

"""Soft radial shell encoder for absorber-centric environment embedding."""

import torch
import torch.nn as nn


def init_mlp_weights(module: nn.Module) -> None:
    """Apply Kaiming-normal initialization to a ``Linear`` layer.

    Args:
        module: Module to initialize; only ``nn.Linear`` instances are affected.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SoftRadialShellsEncoder(nn.Module):
    """Absorber-centric soft-binning over distance with learnable shell centres and widths.

    For each learnable radial shell, computes Gaussian weights over neighbor
    atoms from the absorber-centric distance distribution. These shell-wise
    neighbor weights are used to form a weighted average descriptor per shell;
    the shell summaries are then concatenated and fused with the absorber's own
    descriptor to produce a fixed-size latent vector.

    An optional gating mechanism modulates the fused representation using Fourier
    features of the distance distribution, giving the model flexibility to scale
    contributions depending on the local coordination environment.

    Args:
        d_input: Descriptor feature dimension ``H``.
        n_shells: Number of learnable radial shells.
        latent_dim: Output latent dimension.
        max_radius_angs: Radial cutoff in **A**; neighbors beyond this distance are masked out.
        init_width: Initial Gaussian shell width in **A**.
        use_gating: If ``True``, modulate the shell summary with Fourier features
            of the distance distribution.
    """

    def __init__(
        self,
        d_input: int,
        n_shells: int,
        latent_dim: int,
        max_radius_angs: float,
        init_width: float,
        use_gating: bool,
    ) -> None:
        """Initialize ``SoftRadialShellsEncoder``."""
        super().__init__()
        self.max_radius = float(max_radius_angs)
        self.n_shells = int(n_shells)
        self.d_input = int(d_input)
        self.latent_dim = int(latent_dim)

        # Learnable shell centres (evenly spaced) and widths
        centers = torch.linspace(0.5, self.max_radius - 0.5, steps=self.n_shells)
        widths = torch.full((self.n_shells,), float(init_width))

        self.shell_centers = nn.Parameter(centers)
        self.shell_widths = nn.Parameter(widths.clamp_min(1e-2))

        self.post_shell = nn.Linear(d_input * self.n_shells, d_input)

        # Optional gating using Fourier features of distance distribution
        self.use_gating = bool(use_gating)
        if self.use_gating:
            n_fourier = 8
            self.gate = nn.Sequential(
                nn.Linear(d_input + 2 * n_fourier, d_input),
                nn.GELU(),
                nn.Linear(d_input, d_input),
                nn.Sigmoid(),
            )
            self.register_buffer("freqs", torch.linspace(0.5, 6.0, n_fourier))

        # Fuse absorber + shell summary into latent
        self.fuse = nn.Sequential(
            nn.Linear(d_input * 2, 2 * d_input),
            nn.GELU(),
            nn.Linear(2 * d_input, latent_dim),
        )
        self.apply(init_mlp_weights)

    def _soft_assign(self, r: torch.Tensor) -> torch.Tensor:
        """Gaussian soft assignment of distances to shells.

        Args:
            r: Context distances, shape ``(B, N_ctx)``.

        Returns:
            Per-shell soft weights of shape ``(B, N_ctx, n_shells)``.
            Weights are normalized to sum to 1 along the neighbor dimension.
        """
        centers = self.shell_centers.view(1, 1, -1)
        widths = self.shell_widths.view(1, 1, -1)
        z = (r.unsqueeze(-1) - centers) / (widths + 1e-6)
        w = torch.exp(-0.5 * z * z)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        return w

    def _fourier_feats(self, r: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute mean Fourier features over neighbors.

        Args:
            r: Context distances, shape ``(B, N_ctx)``.
            mask: Optional valid-neighbor mask of shape ``(B, N_ctx)``.
                When given, only masked-in neighbors contribute to the mean.

        Returns:
            Mean Fourier features of shape ``(B, 2 * n_fourier)``.
        """
        f = self.freqs.view(1, 1, -1)
        fsin = torch.sin(r.unsqueeze(-1) * f)
        fcos = torch.cos(r.unsqueeze(-1) * f)
        feats = torch.cat([fsin, fcos], dim=-1)
        if mask is None:
            return feats.mean(dim=1)

        weights = mask.unsqueeze(-1).to(feats.dtype)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        return (feats * weights).sum(dim=1) / denom

    def forward(
        self,
        x: torch.Tensor,
        dists: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the local chemical environment into a fixed-size latent vector.

        Args:
            x: Descriptor features with absorber at index 0, shape ``(B, N, H)``.
            dists: Distances from the absorber atom in **A**, shape ``(B, N)``.
            lengths: Number of real atoms per sample before padding, shape ``(B,)``.
                If ``None``, all positions are treated as real.

        Returns:
            Fused latent representation of shape ``(B, latent_dim)``.
        """
        B, N, H = x.shape
        absorbing = x[:, 0, :]  # (B, H)
        context = x[:, 1:, :]  # (B, N-1, H)
        raw_r = dists[:, 1:]  # (B, N-1)
        r = raw_r.clamp_max(self.max_radius)  # (B, N-1)

        # Build mask for valid context atoms
        if lengths is not None:
            n_ctx = context.size(1)
            idxs = torch.arange(n_ctx, device=x.device)[None, :]
            real_ctx = torch.clamp(lengths - 1, min=0)
            mask = (idxs < real_ctx[:, None]).float()
        else:
            mask = torch.ones(context.shape[:2], device=x.device)
        mask = mask * (raw_r <= self.max_radius).float()

        # Build shell-wise neighbor weights and compute weighted means.
        w = self._soft_assign(r)  # (B, N-1, n_shells)
        w = w * mask.unsqueeze(-1)
        wsum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w = w / wsum

        shell_means = torch.einsum("bns,bnh->bsh", w, context)  # (B, n_shells, H)
        shell_means = shell_means.reshape(B, self.n_shells * H)
        shell_summary = self.post_shell(shell_means)  # (B, H)

        # Optional gating
        if self.use_gating:
            crowd = self._fourier_feats(r, mask=mask)  # (B, 2*n_fourier)
            gate_in = torch.cat([absorbing, crowd], dim=-1)
            g = self.gate(gate_in)
            shell_summary = shell_summary * g

        fused = torch.cat([absorbing, shell_summary], dim=-1)  # (B, 2*H)
        return self.fuse(fused)  # (B, latent_dim)
