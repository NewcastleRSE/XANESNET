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
from e3nn import o3


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
        d = x.shape[-1]
        x = x.reshape(-1, d)

        outs: list[torch.Tensor] = []
        offset = 0
        bflat = x.shape[0]

        for mul, ir in self.irreps:
            dim = ir.dim
            block_dim = mul * dim
            xb = x[:, offset : offset + block_dim].reshape(bflat, mul, dim)

            if ir.l == 0:
                mean = xb.mean(dim=1, keepdim=True)
                var = ((xb - mean) ** 2).mean(dim=1, keepdim=True)
                xb = (xb - mean) / torch.sqrt(var + self.eps)
            else:
                norm = torch.sqrt((xb**2).mean(dim=2, keepdim=True) + self.eps)
                xb = xb / norm

            outs.append(xb.reshape(bflat, block_dim))
            offset += block_dim

        out = torch.cat(outs, dim=-1)

        if self.affine:
            out = out * self.weight + self.bias  # TODO bias equivariance breaking?

        return out.view(*orig_shape, d)
