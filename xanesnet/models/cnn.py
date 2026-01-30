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

from xanesnet.models.base_model import Model
from xanesnet.registry import register_model, register_scheme
from xanesnet.utils.switch import ActivationSwitch


@register_model("cnn")
@register_scheme("cnn", scheme_name="nn")
class CNN(Model):
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
        in_features: int,
        out_features: int,
        hidden_size: int = 256,
        dropout: float = 0.2,
        num_conv_layers: int = 3,
        activation: str = "prelu",
        out_channel: int = 32,
        channel_mul: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        """
        Args:
            in_features (int): Size of input data.
            out_features (int): Size of output data.
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
        self.nn_flag = 1

        # Save model configuration
        self.register_config(locals(), type="cnn")

        act_fn = ActivationSwitch().get(activation)

        # --- Convolutional Layers ---
        conv_layers = []
        in_channel = 1
        current_out_channel = out_channel
        for _ in range(num_conv_layers):
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

        # --- Dense Layers ---
        conv_out_size = self._get_conv_output_size(in_features)

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_features),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        out = self.dense_layers(x)

        return out
