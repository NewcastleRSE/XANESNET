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

from xanesnet.models.base_model import Model
from xanesnet.registry import register_model, register_scheme
from xanesnet.utils.switch import ActivationSwitch


@register_model("mh_cnn")
@register_scheme("mh_cnn", scheme_name="mh")
class MultiHead_CNN(Model):
    """
    A class for constructing a customisable CNN (Convolutional Neural Network) model.
    The model consists of a set of convolutional layers and two dense layers.
    Each convolutional layer comprises a 1D convolution, batch normalisation,
    an activation function, a dropout layer.

    The Number of output channels in the convolutional layers (out_channels) is increased
    multiplicatively at each layer, and it's controlled by a multiplication factor (channel_mul).
    """

    def __init__(
        self,
        in_size: int,
        out_size: List[int],
        hidden_size: int = 256,
        dropout: float = 0.2,
        num_conv_layers: int = 3,
        activation: str = "silu",
        out_channel: int = 32,
        channel_mul: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        head_num_hidden_layers: int = 2,
        head_hidden_size: int = 512,
        head_shrink_rate: int = 1.0,
    ):
        """
        Args:
            in_size (int): Size of input data.
            out_size (int): Size of output data.
            hidden_size (int): Size of the hidden layer in the dense predictor.
            dropout (float): Dropout rate for regularization.
            num_conv_layers (int): Number of convolutional layers in the encoder.
            activation (str): Name of activation function for all layers.
            out_channel (int): Number of output channels for the first conv layer.
            channel_mul (int): Multiplies the number of channels at each subsequent layer.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for convolution and upsampling.
        """
        super().__init__()
        self.mh_flag = 1

        # Save model configuration
        self.register_config(locals(), type="mh_cnn")

        act_fn = ActivationSwitch().get(activation)

        # --- Convolutional Layers ---
        conv_layers = []
        in_channel = 1
        current_out_channel = out_channel
        for i in range(num_conv_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, current_out_channel, kernel_size, stride),
                    nn.BatchNorm1d(current_out_channel),
                    act_fn,
                    nn.Dropout(p=dropout),
                )
            )
            in_channel = current_out_channel
            current_out_channel *= channel_mul

        self.conv_layers = nn.Sequential(*conv_layers)

        conv_out_size = self._get_conv_output_size(in_size)

        # --- Multi-Headed Predictor ---
        self.heads = nn.ModuleList(
            [
                CNNHead(
                    in_size=conv_out_size,
                    out_size=out,
                    num_hidden_layers=head_num_hidden_layers,
                    hidden_size=head_hidden_size,
                    shrink_rate=head_shrink_rate,
                    dropout=dropout,
                    activation=activation,
                )
                for out in out_size
            ]
        )

    def _get_conv_output_size(self, in_size: int) -> int:
        """
        Calculates the output feature dimension of the conv layers by performing
        a single dummy forward pass.
        """
        # Create a dummy input tensor
        dummy_input = torch.randn(1, 1, in_size)

        with torch.no_grad():
            output = self.conv_layers(dummy_input)

        return output.numel()

    def forward(self, x: torch.Tensor, active_head_idx: int = None) -> torch.Tensor:
        x = x.unsqueeze(1)
        shared = self.conv_layers(x)
        shared = torch.flatten(shared, 1)

        if active_head_idx is None:
            # If no specific head is requested, return the stacked output of all heads
            return torch.stack([head(shared) for head in self.heads], dim=0)
        else:
            # Otherwise, return the output of the specified head
            return self.heads[active_head_idx](shared)


class CNNHead(nn.Module):
    """
    A class for constructing a customisable CNN head. This module
    is a dense predictor that takes the flattened output of a CNN's
    convolutional layers and produces a final prediction.

    It consists of a set of hidden layers, each with a linear layer,
    batch normalization, dropout, and an activation function. The final
    layer is a linear layer followed by a Softplus activation.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
        num_hidden_layers: int = 2,
        shrink_rate: float = 1.0,
        activation: str = "silu",
    ):
        """
        Args:
            in_size (int): Size of input data (flattened conv output).
            out_size (int): Size of the final output.
            hidden_size (int): Size of the initial hidden layer.
            dropout (float): Dropout probability for hidden layers.
            num_hidden_layers (int): Number of hidden layers.
            shrink_rate (float): Rate to reduce the hidden layer size multiplicatively.
            activation (str): Name of activation function for hidden layers.
        """
        super().__init__()
        self.mh_flag = 1

        act_fn = ActivationSwitch().get(activation)
        layers = []
        current_size = in_size

        # --- Hidden Layers for the Head ---
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate**i))
            if next_size < 1:
                raise ValueError(
                    f"Head hidden layer {i + 1} size is less than 1. Adjust hidden_size or shrink_rate."
                )

            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.Dropout(dropout))
            layers.append(act_fn)
            current_size = next_size

        # --- Final Output Layer for the Head ---
        layers.append(nn.Linear(current_size, out_size))
        layers.append(nn.Softplus())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
