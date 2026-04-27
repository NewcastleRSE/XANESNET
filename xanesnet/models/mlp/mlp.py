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
from torch import nn

from xanesnet.components import ActivationRegistry, BiasInitRegistry, WeightInitRegistry
from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry


@ModelRegistry.register("mlp")
class MLP(Model):
    """
    A class for constructing a customisable MLP (Multi-Layer Perceptron) model.
    The model consists of a set of hidden layers. All the layers expect the final
    layer, are comprised of a linear layer, a dropout layer, and an activation function.
    The final (output) layer is a linear layer.

    The size of each hidden linear layer is determined by the initial dimension
    (hidden_size) and a reduction factor (shrink_rate) that reduces the layer
    dimension multiplicatively.
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
        """
        Args:
            model_type (str): Model type identifier
            in_size (integer): Size of input data
            out_size (integer): Size of output data
            hidden_size (integer): Size of the initial hidden layer.
            dropout (float): Dropout probability for hidden layers.
            num_hidden_layers (int): Number of hidden layers, excluding input and output layers
            shrink_rate (float): Rate to reduce the hidden layer size multiplicatively.
            activation (str): Name of activation function for hidden layers.
        """
        super().__init__(model_type)

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.shrink_rate = shrink_rate
        self.activation = activation

        act_fn = ActivationRegistry.get(activation)
        layers: list[nn.Module] = []

        # Initialise input and hidden layers
        current_size = in_size
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate**i))
            if next_size < 1:
                raise ValueError(f"Hidden layer {i + 1} size is less than 1. Adjust hidden_size or shrink_rate.")

            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.Dropout(dropout))
            layers.append(act_fn)
            current_size = next_size

        # Initialise output layer
        layers.append(nn.Linear(current_size, out_size))

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dense_layers(x)
        return out

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
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
        """
        Return model signature as a dictionary.
        """
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
