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

"""Scaled-SiLU, dense, and residual layer building blocks for GemNet-OC."""

import math

import torch

from ..utils import he_orthogonal_init


class ScaledSiLU(torch.nn.Module):
    """SiLU activation scaled by ``1 / 0.6`` to preserve unit variance."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled SiLU activation.

        Args:
            x: Input tensor, arbitrary shape.

        Returns:
            Activated tensor with the same shape as ``x``.
        """
        return self._activation(x) * self.scale_factor


class Dense(torch.nn.Module):
    """Linear layer with optional activation and He-orthogonal initialisation.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: Whether to include a bias term.
        activation: Activation function name. Supported: ``"silu"`` /
            ``"swish"``, or ``None`` for identity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: str | None = None,
    ) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["silu", "swish"]:
            self._activation: torch.nn.Module = ScaledSiLU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Activation function {activation!r} not implemented for GemNet-OC.")

    def reset_parameters(self, initializer=he_orthogonal_init) -> None:
        """Re-initialise weights; zero the bias if present.

        Args:
            initializer: Weight initialisation callable. Defaults to
                :func:`~xanesnet.models.gemnet_oc.utils.he_orthogonal_init`.
        """
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transform and optional activation.

        Args:
            x: Input tensor, shape ``(*, in_features)``.

        Returns:
            Output tensor, shape ``(*, out_features)``.
        """
        return self._activation(self.linear(x))


class ResidualLayer(torch.nn.Module):
    """Stack of dense layers with a skip connection scaled by ``1 / sqrt(2)``.

    Args:
        units: Embedding dimension (same for input and output).
        nLayers: Number of dense layers in the residual block.
        layer: Layer constructor. Default: :class:`Dense`.
        **layer_kwargs: Extra keyword arguments forwarded to ``layer``.
    """

    def __init__(self, units: int, nLayers: int = 2, layer=Dense, **layer_kwargs) -> None:
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[layer(in_features=units, out_features=units, bias=False, **layer_kwargs) for _ in range(nLayers)]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor, shape ``(*, units)``.

        Returns:
            Output tensor, shape ``(*, units)``.
        """
        y = self.dense_mlp(x)
        return (x + y) * self.inv_sqrt_2
