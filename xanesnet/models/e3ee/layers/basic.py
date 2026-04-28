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

"""Primitive building blocks (RBF, cutoffs, MLP, IrrepNorm) shared across E3EE layers."""

import torch
import torch.nn as nn
from e3nn import o3


class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion.

    Args:
        start: Left edge of the RBF center grid.
        stop: Right edge of the RBF center grid.
        n_rbf: Number of radial basis functions.
        gamma: Width parameter ``1 / sigma^2``. Defaults to ``1 / delta^2``
            where ``delta`` is the uniform spacing between centers.
    """

    def __init__(self, start: float, stop: float, n_rbf: int, gamma: float | None = None) -> None:
        """Initialize ``GaussianRBF``."""
        super().__init__()
        centers = torch.linspace(start, stop, n_rbf)
        self.register_buffer("centers", centers)
        if gamma is None:
            delta = (stop - start) / max(n_rbf - 1, 1)
            gamma = 1.0 / (delta * delta + 1e-12)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand scalar inputs into Gaussian RBF features.

        Args:
            x: Input scalars of arbitrary leading shape ``(...)``.

        Returns:
            RBF features of shape ``(..., n_rbf)``.
        """
        return torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope.

    Smoothly decays from 1 at ``r = 0`` to 0 at ``r = cutoff``.

    Args:
        cutoff: Cutoff radius in **Angstrom** beyond which the envelope is zero.
    """

    def __init__(self, cutoff: float) -> None:
        """Initialize ``CosineCutoff``."""
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate cosine envelope.

        Args:
            r: Interatomic distances of arbitrary shape ``(...)`` in **Angstrom**.

        Returns:
            Envelope values in ``[0, 1]`` with the same shape as ``r``.
        """
        x = r / self.cutoff
        out = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        out = out * (r <= self.cutoff).to(r.dtype)
        return out


class EnergyRBFEmbedding(nn.Module):
    """Gaussian RBF embedding for scalar energy grid indices.

    Args:
        e_min: Minimum grid index (left edge of RBF center grid).
        e_max: Maximum grid index (right edge of RBF center grid).
        n_rbf: Number of radial basis functions.
    """

    def __init__(self, e_min: float, e_max: float, n_rbf: int) -> None:
        """Initialize ``EnergyRBFEmbedding``."""
        super().__init__()
        self.rbf = GaussianRBF(e_min, e_max, n_rbf)

    def forward(self, energies: torch.Tensor) -> torch.Tensor:
        """Embed energy grid indices as Gaussian RBF features.

        Args:
            energies: Energy grid indices of shape ``(nE,)``.

        Returns:
            RBF features of shape ``(nE, n_rbf)``.
        """
        return self.rbf(energies)


class MLP(nn.Module):
    """Simple multi-layer perceptron with SiLU activations.

    Args:
        in_dim: Size of the input features.
        hidden_dim: Size of each hidden layer.
        out_dim: Size of the output features.
        n_layers: Total number of linear layers (including the output layer).
        dropout: Dropout probability applied after each hidden activation.
            Set to ``0.0`` to disable.
        layer_norm: If ``True``, apply :class:`~torch.nn.LayerNorm` after
            each hidden linear layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        """Initialize ``MLP``."""
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
        """Apply the MLP.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.

        Returns:
            Output tensor of shape ``(..., out_dim)``.
        """
        return self.net(x)


class RadialMLP(nn.Module):
    """Two-hidden-layer MLP for radial (TP) weight prediction.

    Args:
        in_dim: Input size (typically RBF dimension plus optional extras).
        hidden_dim: Size of each hidden layer.
        out_dim: Output size (typically the number of TP weights).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize ``RadialMLP``."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict radial weights.

        Args:
            x: Input features of shape ``(..., in_dim)``.

        Returns:
            Predicted weights of shape ``(..., out_dim)``.
        """
        return self.net(x)


class IrrepNorm(nn.Module):
    """
    Irrep-respecting normalization.

    For ``l = 0`` blocks the multiplicity channels are mean/variance
    normalized.  For ``l > 0`` blocks each irrep copy is RMS-normalized across its irrep dimension.

    Args:
        irreps: The irreps specification that describes the feature layout.
        eps: Small constant added to denominators for numerical stability.
        affine: If ``True``, learn per-channel scale and bias parameters.
    """

    def __init__(self, irreps: o3.Irreps, eps: float = 1e-8, affine: bool = True) -> None:
        """Initialize ``IrrepNorm``."""
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
        """Apply irrep-respecting normalization.

        Args:
            x: Input features of shape ``(..., irreps.dim)``.

        Returns:
            Normalized features with the same shape as ``x``.
        """
        orig_shape = x.shape[:-1]
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)

        outs: list[torch.Tensor] = []
        offset = 0
        bflat = x_flat.shape[0]

        for mul, ir in self.irreps:
            dim = ir.dim
            block_dim = mul * dim
            xb = x_flat[:, offset : offset + block_dim].reshape(bflat, mul, dim)

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
