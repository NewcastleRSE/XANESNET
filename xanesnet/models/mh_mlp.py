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

from typing import List

import torch
from torch import nn

from xanesnet.registry import register_model, register_scheme
from xanesnet.models.base_model import Model
from xanesnet.utils.switch import ActivationSwitch


@register_model("mh_mlp")
@register_scheme("mh_mlp", scheme_name="mh")
class MultiHead_MLP(Model):
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
        in_features: int,
        out_features: List[int],
        hidden_size: int = 512,
        dropout: float = 0.2,
        num_hidden_layers: int = 3,
        shrink_rate: float = 1.0,
        activation: str = "silu",
        head_num_hidden_layers: int = 2,
        head_hidden_size: int = 512,
        head_shrink_rate: int = 1.0,
    ):
        super().__init__()
        self.mh_flag = 1

        # Save model configuration
        self.register_config(locals(), type="mh_mlp")

        # Instantiate ActivationSwitch for dynamic activation selection
        act_fn = ActivationSwitch().get(activation)
        layers = []

        # --- Input and hidden Layers ---
        current_size = in_features
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate**i))
            if next_size < 1:
                raise ValueError(
                    f"Hidden layer {i + 1} size is less than 1. Adjust hidden_size or shrink_rate."
                )

            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.Dropout(dropout))
            layers.append(act_fn)
            current_size = next_size

        # Construct the dense layers as a sequential module by
        # combining all the individual layers created earlier
        self.dense_layers = nn.Sequential(*layers)

        self.heads = nn.ModuleList(
            [
                MLPHead(
                    in_size=current_size,
                    out_size=out,
                    num_hidden_layers=head_num_hidden_layers,
                    hidden_size=head_hidden_size,
                    shrink_rate=head_shrink_rate,
                    dropout=dropout,
                    activation=activation,
                )
                for out in out_features
            ]
        )

    def forward(self, x: torch.Tensor, active_head_idx: int = None) -> torch.Tensor:
        # Feed forward through dense layers
        shared = self.dense_layers(x)
        if active_head_idx is None:
            return torch.stack([head(shared) for head in self.heads], dim=0)
        else:
            return self.heads[active_head_idx](shared)


class MLPHead(nn.Module):
    """
    A class for constructing a customisable MLP (Multi-Layer Perceptron) model that
    is used as one head of a larger multi-headed MLP network. The model consists of
    a set of hidden layers. All the layers expect the final layer, are comprised of
    a linear layer, a dropout layer, and an activation function. The final (output)
    layer is a linear layer.

    The size of each hidden linear layer is determined by the input dimension
    (input_size) and the output dimension (output_size) that reduces the layer
    dimension multiplicatively.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
        num_hidden_layers: int = 3,
        shrink_rate: float = 1.0,
        activation: str = "relu",
    ):
        """
        Args:
            in_size (integer): Size of input data
            out_size (integer): Size of output data
            hidden_size (integer): Size of the initial hidden layer.
            dropout (float): Dropout probability for hidden layers.
            num_hidden_layers (int): Number of hidden layers, excluding input and output layers
            shrink_rate (float): Rate to reduce the hidden layer size multiplicatively.
            activation (str): Name of activation function for hidden layers.
        """

        super().__init__()

        act_fn = ActivationSwitch().get(activation)

        layers = []
        current_size = in_size
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate**i))
            if next_size < 1:
                raise ValueError(
                    f"Hidden layer {i + 1} size is less than 1. Adjust hidden_size or shrink_rate."
                )

            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.Dropout(dropout))
            layers.append(act_fn)
            current_size = next_size

        # Final output layer
        layers.append(nn.Sequential(nn.Linear(current_size, out_size), nn.Softplus()))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
