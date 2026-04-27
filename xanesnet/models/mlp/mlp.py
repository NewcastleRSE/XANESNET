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

"""Multi-layer perceptron (MLP) model for spectroscopy prediction."""

import torch
from torch import nn

from xanesnet.components import ActivationRegistry, BiasInitRegistry, WeightInitRegistry
from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry


@ModelRegistry.register("mlp")
class MLP(Model):
    """A customisable multi-layer perceptron (MLP) for spectroscopy prediction.

    Consists of a sequence of hidden layers followed by a linear output layer. Each hidden
    layer contains a linear transformation, dropout, and an activation function. The final
    layer is a plain linear layer with no activation.

    The hidden layer width starts at ``hidden_size`` and shrinks multiplicatively by
    ``shrink_rate`` at each successive depth step.

    Args:
        model_type: Model type identifier string.
        in_size: Number of input features.
        out_size: Number of output features.
        hidden_size: Width of the first hidden layer.
        dropout: Dropout probability applied after each hidden linear layer. Range ``[0, 1)``.
        num_hidden_layers: Number of hidden layers (excluding the output layer).
        shrink_rate: Multiplicative factor applied to the layer width at each depth step.
        activation: Name of the activation function for hidden layers.
    """

    def __init__(
        self,
        model_type: str,
        # params:
        in_size: int,
        out_size: int,
        hidden_size: int,
        dropout: float,
        num_hidden_layers: int,
        shrink_rate: float,
        activation: str,
    ) -> None:
        super().__init__(model_type)

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.shrink_rate = shrink_rate
        self.activation = activation

        layers: list[nn.Module] = []

        # Initialise input and hidden layers
        current_size = in_size
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate**i))
            if next_size < 1:
                raise ValueError(f"Hidden layer {i + 1} size is less than 1. Adjust hidden_size or shrink_rate.")

            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.Dropout(dropout))
            layers.append(ActivationRegistry.get(activation))
            current_size = next_size

        # Initialise output layer
        layers.append(nn.Linear(current_size, out_size))

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the MLP.

        Args:
            x: Input tensor. ``(batch_size, in_size)``

        Returns:
            Output tensor. ``(batch_size, out_size)``
        """
        return self.dense_layers(x)

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """Initialise all linear layer weights and biases.

        Args:
            weights_init: Name of the weight initialisation scheme (looked up via
                ``WeightInitRegistry``).
            bias_init: Name of the bias initialisation scheme (looked up via
                ``BiasInitRegistry``).
            **kwargs: Extra keyword arguments forwarded to the weight initialiser.
        """
        weight_init_fn = WeightInitRegistry.get(weights_init, **kwargs)
        bias_init_fn = BiasInitRegistry.get(bias_init)

        def _init_layer(m: nn.Module) -> None:
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                weight_init_fn(m.weight)
                assert m.bias is not None, "Bias is None, cannot initialize."
                bias_init_fn(m.bias)

        # Apply to all modules
        self.apply(_init_layer)

    @property
    def signature(self) -> Config:
        """Return the model signature as a :class:`~xanesnet.serialization.config.Config`."""
        signature = super().signature
        signature.update_with_dict(
            {
                "in_size": self.in_size,
                "out_size": self.out_size,
                "hidden_size": self.hidden_size,
                "dropout": self.dropout,
                "num_hidden_layers": self.num_hidden_layers,
                "shrink_rate": self.shrink_rate,
                "activation": self.activation,
            }
        )
        return signature
