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

from typing import Tuple, List

import torch

from torch import nn

from xanesnet.registry import register_model, register_scheme
from xanesnet.models.base_model import Model
from xanesnet.utils.switch import ActivationSwitch


@register_model("mlp")
@register_scheme("mlp", scheme_name="nn")
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
        in_shape: List[int],
        out_shape: List[int],
        hidden_size: int = 256,
        dropout: float = 0.1,
        num_hidden_layers: int = 3,
        shrink_rate: float = 1.0,
        activation: str = "relu",
    ) -> None:
        """
        Args:
            in_shape (List): Shape of input data
            out_shape (List): Shape of output data
            hidden_size (integer): Size of the initial hidden layer.
            dropout (float): Dropout probability for hidden layers.
            num_hidden_layers (int): Number of hidden layers, excluding input and output layers
            shrink_rate (float): Rate to reduce the hidden layer size multiplicatively.
            activation (str): Name of activation function for hidden layers.
        """

        super().__init__()

        self.nn_flag = 1

        # Save model configuration
        self.register_config(locals(), type="mlp")

        in_dim: int = in_shape[0]  # in_shape = (N,)
        out_dim: int = out_shape[0]  # out_shape = (M,)

        act_fn = ActivationSwitch().get(activation)
        layers = []

        # --- Input and hidden Layers ---
        current_size = in_dim
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate**i))
            if next_size < 1:
                raise ValueError(
                    f"Hidden layer {i + 1} size is less than 1. Adjust hidden_size or shrink_rate."
                )

            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.Dropout(dropout))
            layers.append(act_fn)
            current_size = next_size

        # --- Output Layer ---
        layers.append(nn.Linear(current_size, out_dim))

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dense_layers(x)
        return out
