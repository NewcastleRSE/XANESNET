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


@register_model("lstm")
@register_scheme("lstm", scheme_name="nn")
class LSTM(Model):
    """
    A class for constructing a customisable LSTM (Long Short-Term Memory) model.
    The model consists of a bidirectional LSTM layer follow by a
    feedforward neural network with two dense layers.

    The defined LSTM is bidirectional, processing the input sequence in both
    forward and backward directions. The output size of the LSTM is
    2 x the number of features in the LSTM hidden state (hidden_size).
    The intermediate hidden size of the following FNN is user-specified
    (hidden_out_size)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int = 256,
        hidden_out_size: int = 128,
        num_layers: int = 5,
        dropout: float = 0.2,
        activation: str = "prelu",
    ):
        """
        Args:
            in_features (integer): Input size
            out_features (integer): Output size
            hidden_size (integer): Number of features in the hidden state of
                the LSTM layer.
            hidden_out_size (integer): Intermediate size of the two dense layers
                in the feedforward neural network.
            num_layers (integer): Number of recurrent layers (LSTM blocks)
                in the network.
            dropout (float): If none-zero, add a dropout layer on the outputs
                of the dense layer, with dropout probability equal to dropout.
            activation (string): Name of activation function for
                the dense layers.
        """
        super().__init__()

        self.nn_flag = 1

        # Save model configuration
        self.register_config(locals(), type="lstm")

        act_fn = ActivationSwitch().get(activation)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_out_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_out_size, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        out = self.dense_layers(x)
        return out
