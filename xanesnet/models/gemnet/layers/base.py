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

from ..utils import he_orthogonal_init


class Dense(torch.nn.Module):
    """
    Combines dense layer and scaling for swish activation.

    Parameters
    ----------
        in_features: int
            Input size.
        out_features: int
            Output size.
        bias: bool
            True if use bias.
        activation: str | None
            Name of the activation function to use.

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
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown activation function '{activation}' specified for Dense layer.")

    def reset_parameters(self) -> None:
        he_orthogonal_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self._activation(x)
        return x


class ScaledSiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._activation(x) * self.scale_factor


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        activation: str
            Name of the activation function to use.
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

    def reset_parameters(self) -> None:
        for layer in self.dense_mlp:
            layer.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dense_mlp(inputs)
        x = inputs + x
        x = x * self.inv_sqrt_2
        return x
