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

"""Base dense and residual layer building blocks for GemNet."""

import torch

from ..utils import he_orthogonal_init


class Dense(torch.nn.Module):
    """Dense (linear) layer with optional scaled SiLU activation.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: If ``True``, include a learnable bias term.
        activation: Activation function name. Supported values are
            ``"swish"`` / ``"silu"`` (scaled SiLU) and ``None`` (identity).
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
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        if isinstance(activation, str):
            activation = activation.lower()
        self._activation: ScaledSiLU | torch.nn.Identity
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown activation function '{activation}' specified for Dense layer.")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise linear weights with He-orthogonal init and zero bias."""
        he_orthogonal_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transform followed by the activation.

        Args:
            x: Input features of shape ``(*, in_features)``.

        Returns:
            Output features of shape ``(*, out_features)``.
        """
        x = self.linear(x)
        x = self._activation(x)
        return x


class ScaledSiLU(torch.nn.Module):
    """SiLU (Swish) activation scaled by ``1 / 0.6`` for variance preservation."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled SiLU activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor scaled by ``1 / 0.6``.
        """
        return self._activation(x) * self.scale_factor


class ResidualLayer(torch.nn.Module):
    """Residual block with output scaled by ``1 / sqrt(2)``.

    Args:
        units: Feature dimension (input = output).
        activation: Activation function name (forwarded to :class:`Dense`).
        nLayers: Number of :class:`Dense` layers in the residual branch.
    """

    def __init__(
        self,
        units: int,
        activation: str,
        nLayers: int = 2,
    ) -> None:
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[Dense(units, units, activation=activation, bias=False) for _ in range(nLayers)]
        )
        self.inv_sqrt_2 = 1 / (2.0**0.5)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise all inner :class:`Dense` layers."""
        for layer in self.dense_mlp:
            layer.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the residual transformation.

        Args:
            inputs: Input features of shape ``(*, units)``.

        Returns:
            Output features of shape ``(*, units)``, scaled by ``1 / sqrt(2)``.
        """
        x = self.dense_mlp(inputs)
        x = inputs + x
        x = x * self.inv_sqrt_2
        return x
