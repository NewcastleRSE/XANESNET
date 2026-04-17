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


def init_mlp_weights(module: nn.Module) -> None:
    """Kaiming-normal initialisation for Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SoftRadialShellsEncoder(nn.Module):
    """
    Absorber-centric soft-binning over distance with learnable shell centres/widths.

    Assigns each neighbour atom a soft weight to one of ``n_shells`` radial shells
    via a Gaussian kernel centred on learnable positions. The weighted descriptor
    features are averaged per shell, concatenated, and fused with the absorber's
    own descriptor to produce a fixed-size latent vector.

    An optional gating mechanism modulates the fused representation using Fourier
    features of the distance distribution, giving the model flexibility to scale
    contributions depending on the local coordination environment.

    Inputs:
        x:       (B, N, H) descriptor features, absorber at index 0
        dists:   (B, N) distances from absorber
        lengths: (B,) number of real atoms per sample (optional)
    Output:
        (B, latent_dim)
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
        """Gaussian soft assignment of distances to shells. Returns (B, N_ctx, n_shells)."""
        centers = self.shell_centers.view(1, 1, -1)
        widths = self.shell_widths.view(1, 1, -1)
        z = (r.unsqueeze(-1) - centers) / (widths + 1e-6)
        w = torch.exp(-0.5 * z * z)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        return w

    def _fourier_feats(self, r: torch.Tensor) -> torch.Tensor:
        """Mean Fourier features over neighbours. Returns (B, 2*n_fourier)."""
        f = self.freqs.view(1, 1, -1)
        fsin = torch.sin(r.unsqueeze(-1) * f)
        fcos = torch.cos(r.unsqueeze(-1) * f)
        return torch.cat([fsin, fcos], dim=-1).mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        dists: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, N, H) descriptor features; absorber at index 0.
            dists:   (B, N) distances from absorber atom.
            lengths: (B,) number of real atoms per sample (before padding).

        Returns:
            (B, latent_dim) fused latent representation.
        """
        B, N, H = x.shape
        absorbing = x[:, 0, :]  # (B, H)
        context = x[:, 1:, :]  # (B, N-1, H)
        r = dists[:, 1:].clamp_max(self.max_radius)  # (B, N-1)

        # Build mask for valid context atoms
        if lengths is not None:
            n_ctx = context.size(1)
            idxs = torch.arange(n_ctx, device=x.device)[None, :]
            real_ctx = torch.clamp(lengths - 1, min=0)
            mask = (idxs < real_ctx[:, None]).float()
        else:
            mask = torch.ones(context.shape[:2], device=x.device)
        mask = mask * (r <= self.max_radius).float()

        # Soft-assign neighbours to shells and compute weighted means
        w = self._soft_assign(r)  # (B, N-1, n_shells)
        w = w * mask.unsqueeze(-1)
        wsum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w = w / wsum

        shell_means = torch.einsum("bns,bnh->bsh", w, context)  # (B, n_shells, H)
        shell_means = shell_means.reshape(B, self.n_shells * H)
        shell_summary = self.post_shell(shell_means)  # (B, H)

        # Optional gating
        if self.use_gating:
            crowd = self._fourier_feats(r)  # (B, 2*n_fourier)
            gate_in = torch.cat([absorbing, crowd], dim=-1)
            g = self.gate(gate_in)
            shell_summary = shell_summary * g

        fused = torch.cat([absorbing, shell_summary], dim=-1)  # (B, 2*H)
        return self.fuse(fused)  # (B, latent_dim)
